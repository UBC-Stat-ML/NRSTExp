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

# constructor: copy key fields of an existing (usually pre-tuned) AbstractSTSampler
GT95Sampler(st::NRST.AbstractSTSampler) = GT95Sampler(NRST.copyfields(st)...)

###############################################################################
# sampling methods
###############################################################################

#######################################
# communication step
#######################################

function propose_i(gt::GT95Sampler{T,I,K}, rng::AbstractRNG) where {T,I,K}
    @unpack np,ip,curV = gt
    @unpack N,betas,c = np
    i = first(ip)
    if i == zero(I)
        # q_{10}/q_{01} = (1/2)/1 = 1/2 => -log(q_{10}/q_{01}) = log(2), and similarly for i==N
        iprop = one(I)
        nlpr  = logtwo
    elseif i == N
        iprop = N-one(I)
        nlpr  = logtwo
    else
        iprop = i + (rand(rng, Bool) ? -one(I) : one(I))
        # q_{01}/q_{10} = 1/(1/2) = 2 => -log(q_{01}/q_{10}) = -log(2), and similarly for other boundary
        nlpr  = (iprop==zero(I) || iprop==N) ? (-logtwo) : zero(K)
    end
    gt.ip[end] = iprop - i # store direction of movement
    nlar = nlpr + NRST.get_nlar(betas[i+1],betas[iprop+1],c[i+1],c[iprop+1],curV[])
    return iprop, nlar
end
function NRST.comm_step!(gt::GT95Sampler, rng::AbstractRNG)
    iprop,nlar = propose_i(gt,rng)  # updates direction of movement
    (nlar < randexp(rng)) && (gt.ip[begin] = iprop)
    return NRST.nlar_2_rp(nlar)
end

#######################################
# RegenerativeSampler interface
#######################################

# check if state is in the atom
NRST.isinatom(gt::GT95Sampler{T,I}) where {T,I} = (first(gt.ip)==zero(I))

# move state to the atom
NRST.toatom!(gt::GT95Sampler{T,I}) where {T,I} = (gt.ip[1]=zero(I))

# handling last tour step
function NRST.save_last_step_tour!(gt::GT95Sampler{T,I,K}, tr; kwargs...) where {T,I,K}
    NRST.save_pre_step!(gt, tr; kwargs...)                   # store state at atom
    rp = NRST.nlar_2_rp(last(propose_i(gt, TaskLocalRNG()))) # simulate a comm step to get rp. since ip[1]=0, the result is deterministic, so the RNG is not actually used 
    NRST.save_post_step!(gt, tr, rp, K(NaN), one(I))         # the expl step would not use an explorer; thus the NaN. Also, we assume the draw from the reference would succeed, thus using only 1 V(x) eval 
end

###############################################################################
# Tuning for GT95Sampler
###############################################################################

function init_and_tune(
    ::Type{TGT},
    tm::NRST.TemperedModel,
    rng::AbstractRNG;
    N_init     = 5,
    ntours     = 2^9,
    C0::Real   = 1.,
    N0::Real   = 0.,
    tacc::Real = 0.3,
    max_rounds = 5
    ) where {TGT<:GT95Sampler}
    gt = NRST.init_sampler(GT95Sampler, tm, rng, N = N_init, tune=false, nexpl=1)
    copyto!(gt.np.betas, [0.;10. .^ range(-16,-3, N_init)]) # init grid
    ltacc = log(tacc)
    for r in 1:max_rounds
        np = gt.np
        N  = np.N
        println("round $r: N=$N")

        # Step 1
        print("\tStep 1...")
        nsteps = 2 * (N+1) * ntours
        run_and_adjust!(gt, rng, nsteps, C0, N0)
        println("done!")

        # Step 2
        print("\tStep 2...")
        visits, as = run_and_collect!(gt, rng, nsteps)
        os   = vec(sum(visits,dims=2)) # occupation times
        np.c .-= log.(os ./ sum(os))
        println("done!")
        
        # Step 3
        print("\tStep 3...")
        cfun   = CubicSpline(np.betas, np.c)
        las    = log.(as)
        lalpha = sum(las) / N
        print("mean(as) = $( round(exp(lalpha), digits=2) )...")
        adjust_grid(np.betas, las, lalpha)
        map!(cfun, np.c, np.betas)
        println("done!")

        # Step 4
        print("\tStep 4...")
        visits, as = run_and_collect!(gt, rng, nsteps)
        ma = mean(as[(begin+1):(end-1)])
        print("mean(as) = $(round(ma,digits=2))...")
        last(np.betas)>=1. && abs(ma-tacc) < 0.1 && (println("FINISHED!"); break)
        error("Step 4 missing (TODO!!)")
    end
    gt
end

# step 1
function run_and_adjust!(gt::GT95Sampler, rng::AbstractRNG, nsteps, args...)
    for k in 1:nsteps
        step_and_adjust!(gt, rng, k, args...)
    end
end
function step_and_adjust!(gt::GT95Sampler, rng::AbstractRNG, k, C0, N0)
    m = gt.np.N+1
    NRST.step!(gt, rng)
    ip1 = first(gt.ip)+1
    gt.np.c[setdiff(1:m,ip1)] .+= C0/(m*(k+N0))
    gt.np.c[ip1] -= C0/(k+N0)
    return
end

# step 2
function run_and_collect!(
    gt::GT95Sampler{T,TI,TF},
    rng::AbstractRNG, 
    nsteps
    ) where {T,TI<:Int,TF<:AbstractFloat}
    N = gt.np.N
    visits = zeros(TI, (N+1, 2))
    rpacc  = zeros(TF, (N+1, 2))
    for _ in 1:nsteps
        i = first(gt.ip)
        rp, _, _ = NRST.step!(gt, rng)
        ieps = last(gt.ip) > 0 ? 1 : 2
        visits[i+1,ieps] += one(TI)
        rpacc[i+1,ieps]  += rp
    end
    return (visits, aveacc(gt,visits,rpacc))
end
# use instructions at bottom of p4
function aveacc(::GT95Sampler,visits,rpacc)
    A  = 1. .- rpacc ./ visits
    as = (A[begin:(end-1),begin] + A[(begin+1):end,end])/2
    as[begin] += A[begin,begin]/2
    as[end]   += A[end,end]/2
    return as
end

## Step 3
# adjust grid by imposing equal acceptance prob
function adjust_grid(betas,las,lalpha)
    LCA = LogCumAcc(betas,las)
    N   = length(las)
    for i in 1:(N-1)
        lca_target = i*lalpha
        betas[i+1] = inverse_LCA(LCA, lca_target)
    end
    betas
end
# log of cumulative acceptance from a continuous move between 0 and s with no rejs
#     log(A(s)) := -int_0^s ds b(s)
# with b defined in the eq after eq 2. Then
#     log(A(s)) =  log(a_{i(s)}) (s - lam_{i(s)})/(lam_{i(s)+1}-lam_i(s)) + sum_{i=1}^{i(s)-1} log(ai)
# where
#     i(s) := max{i: s > lam_i}
struct LogCumAcc{TF<:AbstractFloat}
    betas::Vector{TF}
    las::Vector{TF}
    clas::Vector{TF}
end
LogCumAcc(betas,las) = LogCumAcc(betas,las,cumsum(las))
function (LCA::LogCumAcc)(s::Real)
    @unpack betas,las,clas = LCA
    @assert Base.isbetween(first(betas), s, last(betas))
    is = searchsortedprevious(betas, s)
    las[is]*(s-betas[is])/(betas[is+1]-betas[is]) + (is>1 ? clas[is-1] : 0.)
end
function inverse_LCA(LCA::LogCumAcc, lca::Real)
    @unpack betas,las,clas = LCA
    @assert Base.isbetween(first(clas), lca, last(clas))
    is = searchsortednext(clas, lca,rev=true)
    betas[is] + (lca - (is>1 ? clas[is-1] : 0.))*(betas[is+1]-betas[is])/las[is]
end
