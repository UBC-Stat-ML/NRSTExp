###############################################################################
# Implements Geyer & Thompson (1995)
# notes on the observed rejection probabilities: under an appropriately tuned
# grid and c's, the Hastings correction has the effect of making it easier to 
# reach the boundary, but difficult to escape it. Experiments show about ~50%
# rejection regardless of the size of the grid. Incidentally, this is what one
# would get by averaging the rp's at the extremes of NRST. 
###############################################################################

# exact same fields as NRSTSampler 
struct GT95Sampler{T,I<:Int,K<:AbstractFloat,TXp<:NRST.ExplorationKernel,TProb<:NRST.NRSTProblem} <: NRST.AbstractSTSampler{T,I,K,TXp,TProb}
    np::TProb              # encapsulates problem specifics
    xpl::TXp               # exploration kernel
    x::T                   # current state of target variable
    ip::MVector{2,I}       # current state of the Index Process (i,eps). uses statically sized but mutable vector
    curV::Base.RefValue{K} # current energy V(x) (stored as ref to make it mutable)
end

# constructor: copy key fields of an existing (usually pre-tuned) NRSTSampler
GT95Sampler(ns::NRSTSampler) = GT95Sampler(NRST.copyfields(ns)...)

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

function propose_i(gt::GT95Sampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack np,ip,curV = gt
    @unpack N,betas,c = np
    i = ip[1]
    if i == zero(I)
        # q_{10}/q_{01} = (1/2)/1 = 1/2 => -log(q_{10}/q_{01}) = log(2), and similarly for i==N
        iprop = one(I)
        nlpr  = logtwo
    elseif i == N
        iprop = N-one(I)
        nlpr  = logtwo
    else
        iprop = i + (rand(rng) < 0.5 ? -one(I) : one(I))
        # q_{01}/q_{10} = 1/(1/2) = 2 => -log(q_{01}/q_{10}) = -log(2), and similarly for other boundary
        nlpr  = (iprop==zero(I) || iprop==N) ? (-logtwo) : zero(K)
    end        
    nlar = nlpr + NRST.get_nlar(betas[i+1],betas[iprop+1],c[i+1],c[iprop+1],curV[])
    return iprop, nlar
end
function NRST.comm_step!(gt::GT95Sampler, rng::AbstractRNG)
    iprop,nlar = propose_i(gt,rng)
    (nlar < randexp(rng)) && (gt.ip[1] = iprop)
    return NRST.nlar_2_rp(nlar)
end

#######################################
# RegenerativeSampler interface
#######################################

# check if state is in the atom
NRST.isinatom(gt::GT95Sampler{T,I}) where {T,I} = (gt.ip[1]==zero(I))

# move state to the atom
NRST.toatom!(gt::GT95Sampler{T,I}) where {T,I} = (gt.ip[1]=zero(I))

# handling last tour step
function NRST.save_last_step_tour!(gt::GT95Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(gt, tr; kwargs...)                   # store state at atom
    rp = NRST.nlar_2_rp(last(propose_i(gt, TaskLocalRNG()))) # simulate a comm step to get rp. since ip[1]=0, the result is deterministic, so the RNG is not actually used 
    NRST.save_post_step!(gt, tr, rp, K(NaN), one(I))         # the expl step would not use an explorer; thus the NaN. Also, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end
