###############################################################################
# Implements Faizi et al. (2020) with Irreversible Metropolized-Gibbs sampler.
# Only the special case δ=1, shown by authors to outperform others
# adversarial case: MRNATrans seed 40322 γ=2.5 maxcor=0.95 SSSO
###############################################################################

# exact same fields as NRSTSampler, plus storage for calculations
struct FBDRSampler{T,I<:Int,K<:AbstractFloat,TXp<:NRST.ExplorationKernel,TProb<:NRST.NRSTProblem} <: NRST.AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
    gs::Vector{K}          # storage for the log of the conditional beta|x, length N+1
    ms::Vector{K}          # storage for the log of the Irreversible Metropolized Gibbs conditionals, length N+1 (eq 31, formula for j neq i)
end

# constructor: copy key fields of an existing (usually pre-tuned) NRSTSampler
function FBDRSampler(ns::NRSTSampler)
    FBDRSampler(NRST.copyfields(ns)...,similar(ns.np.c),similar(ns.np.c))
end

# specialized copy method to deal with extra storage fields
function Base.copy(fbdr::TS) where {TS <: FBDRSampler} 
    TS(NRST.copyfields(fbdr)...,similar(fbdr.np.c),similar(fbdr.np.c))
end

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

# update the log of the Gibbs conditional of beta
#     G_i = exp(-b_iV + c_i)/ sum_j exp(-b_jV + c_j)
#     g_i := log(G_i) = -b_iV + c_i - logsumexp(-bV + c)
# note: it holds that sum(exp, gs) === one(eltype(gs))
function update_gs!(fbdr::FBDRSampler)
    @unpack gs,np,curV = fbdr
    @unpack betas,c = np
    gs  .= -betas .* curV[] .+ c
    gs .-= logsumexp(gs)
end

# update the log of the IMGS conditional of beta for j neq i
# This is Eq 31, case j neq i
#     M_{i,j}^{eps} = 1{(j-i)eps>0} M_{i,j}
# with M_{i,j} the (standard) Metropolized Gibbs (Eq 15, j neq i)
#     M_{i,j} = G_j min{1/(1-G_i), 1/(1-G_j)} = G_j/max{1-G_i, 1-G_j}
# Furthermore, since log is increasing,
#     m_j := log(M_{i,j}) = g_j - log(max{1-exp(g_i), 1-exp(g_j)})= g_j - max{log1mexp(g_i),log1mexp(g_j)}, (j-i)eps>0
# Note: ms has missing mass, so sum(exp, ms)<1.
function update_ms!(fbdr::FBDRSampler{T,I,K}) where {T,I,K}
    update_gs!(fbdr)
    @unpack gs,ms,ip = fbdr
    i   = first(ip)
    idx = i+one(I)
    ϵ   = last(ip)
    log1mexpgi = log1mexp(gs[idx])
    for (jdx,g) in enumerate(gs)
        @inbounds ms[jdx] = sign(jdx-idx)!=sign(ϵ) ? K(-Inf) : g - max(log1mexp(g), log1mexpgi)
    end
end

# full tempering step
function NRST.comm_step!(fbdr::FBDRSampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack ms,ip,np = fbdr
    zK = zero(K)                                     # attempt to move i
    update_ms!(fbdr)                                 # update IMGS probabilities
    idxp = sample_logprob(rng, ms)                   # sample a new idxp = iprop + 1 (1-based index)
    lpff = log1mexp(min(zK, logsumexp(ms)))          # logprob of failing to sample from {j: (j-i)eps>0}: log(1-sum(M_{i,j})) = log(1-exp(logsumexp(m))) = log1mexp(logsumexp(m)). also, need to trunc to avoid issues with numerical noise
    if idxp > zero(I)
        ip[1] = idxp - one(I)                        # correct for 1-based index
    else                                             # failed to sample from {j: (j-i)eps>0} => need to sample from 2 options: {flip, stay}
        ip[2] *= -one(I)                             # simulate flip
        update_ms!(fbdr)                             # recompute IMGS probabilities
        lpfb   = log1mexp(min(zK, logsumexp(ms)))    # logprob of failing to sample from {j: (j-i)eps'>0} with the flipped eps'=-eps. also, need to trunc to avoid issues with numerical noise
        lpflip = log1mexp(min(zK, lpfb-lpff))        # log-probability of flip = log(Lambda) - lpff but more accurate (see below)
        randexp(rng) < -lpflip && (ip[2] *= -one(I)) # flip failed => need to undo ϵ flip   
    end
    return exp(lpff)                                 # return prob of failing to sample {j: (j-i)eps>0}
end

# note on the expression for lpflip. The prob of flip is
# pflip = Lambda / 1 - sum_{l neq k} M_{k,l}^{eps}
# Let
# pfail_k^{eps} := 1 - sum_{l neq k} M_{k,l}^{eps}
# this is the prob of either flipping or not chaning anything; i.e.,
# pfail_k^{eps} = Lambda_k^{eps} + M_{k,k}^{eps}
# (btw this shows more precisely why 2nd line in Eq 31 is wrong)
# Then
# Lambda_k^{eps} = max{0,  1-pfail_k^{-eps}-  1+pfail_k^{eps}}
# = max{0,  pfail_k^{eps} - pfail_k^{-eps}}
# Then, the prob of accepting the flip is
# Lambda_k^{eps} / pfail_k^{eps}
# = (1/pfail_k^{eps})max{0,  pfail_k^{eps} - pfail_k^{-eps}}
# = max{0, 1 - pfail_k^{-eps}/pfail_k^{eps}}
# and moreover, if 
# lpf_k^{eps} := log(pfail_k^{eps})
# Then
# 1 - pfail_k^{-eps}/pfail_k^{eps}
# = 1 - exp[lpf_k^{-eps}  - lpf_k^{eps} ]
# = - (exp[lpf_k^{-eps}  - lpf_k^{eps} ])
# = - expm1[lpf_k^{-eps}  - lpf_k^{eps} ] 
# Furthermore, since expm1 is increasing and expm1(0)=0 
# max{0, -expm1(a)} = -[min{expm1(0), expm1(a)}]
# = -expm1(min{0, a})
# Finally
# Lambda_k^{eps} / pfail_k^{eps}
# = -expm1(min{0, lpf_k^{-eps}  - lpf_k^{eps}  })
# Even better: keep log scale by using
# log(Lambda_k^{eps} / pfail_k^{eps})
# = log[1 - exp(min{0, lpf_k^{-eps}  - lpf_k^{eps}  }) ]
# = log1mexp(min{0, lpf_k^{-eps}  - lpf_k^{eps}  })

# Note: this must return true
# update_ms!(fbdr)
# (last(fbdr.ip) > 0 ? findlast(isinf,fbdr.ms)-1 : findlast(isfinite,fbdr.ms)) == first(fbdr.ip)

#######################################
# RegenerativeSampler interface
#######################################

# check if state is in the atom
function NRST.isinatom(fbdr::FBDRSampler{T,I}) where {T,I}
    fbdr.ip[1]==zero(I) && fbdr.ip[2]==one(I)
end

# move state to the atom
function NRST.toatom!(fbdr::FBDRSampler{T,I}) where {T,I}
    fbdr.ip[1] = zero(I)
    fbdr.ip[2] = one(I)
end

# handling last tour step
function NRST.save_last_step_tour!(fbdr::FBDRSampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(fbdr, tr; kwargs...)           # store state at atom
    update_ms!(fbdr)                                   # update IMGS probabilities
    rp = exp(fbdr.ms[first(fbdr.ip)+1])                # probability of iprop==i <=> prob of rejecting an i move
    NRST.save_post_step!(fbdr, tr, rp, K(NaN), one(I)) # the expl step would not use an explorer; thus the NaN. Also, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end
