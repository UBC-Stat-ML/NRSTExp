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

#######################################
# RegenerativeSampler interface
#######################################

# check if state is in the atom
function NRST.isinatom(sh::SH16Sampler{T,I}) where {T,I}
    sh.ip[1]==zero(I) && sh.ip[2]==one(I)
end

# move state to the atom
function NRST.toatom!(sh::SH16Sampler{T,I}) where {T,I}
    sh.ip[1] = zero(I)
    sh.ip[2] = one(I)
end

# handling last tour step. same as NRST.
function NRST.save_last_step_tour!(sh::SH16Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(sh, tr; kwargs...)               # store state at atom
    NRST.save_post_step!(sh, tr, one(K), K(NaN), one(I)) # move towards 1 from (0,-) is always rejected. Also, it doesn't use an explorer so NaN. Finally, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end

###############################################################################
# Tuning for SH16Sampler from Sakai & Hukushima (2016b)
# Note: for each i∈{0..N}, let mv_{i+1} the running mean at level i. Then
#     c_i     = c_{i+1} + δβ[mv_i + mv_{i+1}]/2
#     c_{i+1} = c_{i+2} + δβ[mv_{i+1} + mv_{i+2}]/2
# So, when level is i<N, the updated mv_{i+1} can be used to update two c's,
# This is how I interpret the phrase in point (d)
#     > Continue the IST simulation, calculate Ē1 and Ē2, and 
#     > update g2 and g3 after every (n) MCS.
# This is obviously wrong on its face since you can only update mv_{i+1} (you 
# only see one i at each step), but you *can* update two (contiguous) cs.
###############################################################################

function NRST.tune!(
    sh::SH16Sampler{T,TI,TF},
    rng::AbstractRNG;
    betas::AbstractVector = range(zero(TF), one(TF), sh.np.N+1), # uses equidistant grid by default
    min_steps::Int = 0,
    max_steps::Int = 1_000_000,
    correct::Bool = false,
    xv_init = nothing,
    min_visits::Int = 1,                                   # min number of visits to i=0
    ) where {T,TI<:Int,TF<:AbstractFloat}
    # tune grid
    np = sh.np
    copyto!(np.betas, betas)
    
    # tune the affinities
    N   = np.N
    hδβ = inv(N)/2                                         # half δβ == (beta-beta')/2
    c   = np.c
    fill!(c, zero(TF))                                     # init c
    mvs = [Mean(TF) for _ in 0:N]                          # init N+1 OnlineStats Mean accumulators for mean Vs
    sh.ip[begin] = N                                       # start simulation from the coldest level == largest beta
    sh.ip[end] = rand(rng, Bool) ? one(TI) : -one(TI)      # select random eps
    if isnothing(xv_init)
        NRST.refreshx!(sh, rng)                            # select random x
    else
        copyto!(sh.x, first(xv_init))
        sh.curV[] = last(xv_init)
    end

    # on-the-fly weight determination loop
    n   = 0
    nv0 = 0
    while n < max_steps && (nv0 < min_visits || n < min_steps)
        n += 1
        _, _ , nvs = NRST.step!(sh, rng)
        i   = first(sh.ip)
        ip1 = i+1
        ip2 = i+2
        i == zero(TI) && (nv0 += 1; println("Step $n: nv0=$(nv0)."))
        fit!(mvs[ip1], -sh.curV[])                         # update running mean of -V at i
        EV1 = value(mvs[ip1])                              # extract the updated value
        if i==N
            c[N] = hδβ*EV1                                 # special update, implicitly assumes c[N+1] = -mv_N*db/2
        else
            i == N-1 && correct && (c[end] = -hδβ*EV1)     # this makes the update for i==N valid but it is not in the paper
            c[ip1] = c[ip2] + hδβ*(value(mvs[ip2]) + EV1)
            if i > zero(TI)
                c[i] = c[ip1] + hδβ*(value(mvs[i]) + EV1)
            end
        end
    end
end

