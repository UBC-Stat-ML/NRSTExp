###############################################################################
# Implements Sakai & Hukushima (2016), only the special case δ=1
###############################################################################

# exact same fields as NRSTSampler 
struct SH16Sampler{T,I<:Int,K<:AbstractFloat,TXp<:NRST.ExplorationKernel,TProb<:NRST.NRSTProblem} <: NRST.AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

# constructor: copy key fields of an existing (usually pre-tuned) NRSTSampler
SH16Sampler(ns::NRSTSampler) = SH16Sampler(NRST.copyfields(ns)...)

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

# Propose i and compute log target ratio, log Hastings ratio, and logprob of moving. 
# This is eqs 12, 13, 14. Note: for δ=1 the proposals are deterministic
#     q_{0,1}^{eps} = q_{N,N-1}^{eps} = 1, at boundaries // this is true for all δ
#     q_{r,l}^{eps} = 1{l=r+eps},          o.w.
# Let
#     W_{r,l}^{eps} := min{1, [q_{l,r}^{-eps}/q_{r,l}^{eps}] [P_l/P_r] }
# Then
#     W_{0,1}^{-} = 0                // since q_{1,0}^+ = 0 but q_{0,1}^-=1 (i.e., well-defined but 0)
#     W_{N,N-1}^{+} = 0              // since q_{N-1,N}^- = 0 but q_{N,N-1}^+=1 (i.e., well-defined but 0)
# Hence
#     W_{r,l}^{eps} = 1{0 <= r+eps <= N}} min{1,P_l/P_r}, o.w.
# Finally,
#     T_{r,l}^{eps} = q_{r,l}^{eps}W_{r,l}^{eps}
function propose_i(sh::SH16Sampler{T,I,K}) where {T,I,K}
    @unpack np,ip,curV = sh
    @unpack N,betas,c = np
    i   = first(ip)
    ϵ   = last(ip)
    lhr = zero(K)  # log Hastings ratio
    if i == zero(I)
        iprop = one(I)
        ϵ < zero(I) && (lhr = K(-Inf))
    elseif i == N
        iprop = N-one(I)
        ϵ > zero(I) && (lhr = K(-Inf))
    else
        iprop = i + ϵ
    end        
    ltr = -NRST.get_nlar(betas[i+1],betas[iprop+1],c[i+1],c[iprop+1],curV[])
    lpm = min(zero(K), lhr + ltr)
    return iprop, lpm
end

# compute log prob of flipping. note: it changes ip!
function get_lpflip!(sh::SH16Sampler{T,I,K}, lpm) where {T,I,K}
    lpff      = log1mexp(lpm)                     # log prob of failing the i move = log(1- exp(lpm))=log1mexp(lpm) 
    sh.ip[2] *= -one(I)                           # simulate flip
    _, lpmb   = propose_i(sh)                     # get prob moving i if we had flipped 
    lpfb      = log1mexp(lpmb)                    # log prob of failing to move i if we had moved in the opp direction
    lpflip    = log1mexp(min(zero(K), lpfb-lpff)) # pflip = max(0, pff - pfb)/pff = max(0, 1 - exp(lpfb-lpff))
    return lpflip
end

# full tempering step. This is point (1) of the algorithm in Sec. 3.2.
function NRST.comm_step!(sh::SH16Sampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    iprop, lpm = propose_i(sh)
    if -lpm < randexp(rng)
        sh.ip[1] = iprop
    else
        lpflip = get_lpflip!(sh, lpm)                   # compute flip probability by simulating a flip
        -lpflip > randexp(rng) && (sh.ip[2] *= -one(I)) # failed, need to undo the simulated flip inside get_lpflip!
    end
    return -expm1(lpm)
end

# same step! method as NRST

#######################################
# RegenerativeSampler interface
#######################################

# same atom as NRST, no need for specialized isinatom method

# reset state by sampling from the renewal measure
# Although W_{0,1}^{-}=0 <=> pfail_0^{-} = 1, it is not clear that pfail_0^{+}=0.
# Therefore, it is not clear that SH16 always flips at (0,-), so safer to have 
# a specialized renew method
function NRST.renew!(sh::SH16Sampler{T,I}, rng::AbstractRNG) where {T,I}
    sh.ip[1] = zero(I)
    sh.ip[2] = -one(I)
    NRST.step!(sh, rng)
end

# handling last tour step. same as NRST.
function NRST.save_last_step_tour!(sh::SH16Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(sh, tr; kwargs...)               # store state at atom
    NRST.save_post_step!(sh, tr, one(K), K(NaN), one(I)) # move towards 1 from (0,-) is always rejected. Also, it doesn't use an explorer so NaN. Finally, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end
