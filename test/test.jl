using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = ChalLogistic();#TitanicHS()#MRNATrans()#XYModel(6)#HierarchicalModel()##XYModel(12)##MRNATrans()##XYModel(8)##
rng = SplittableRandom(5470)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    N=12,
    adapt_N_rounds=0
);

using NRSTExp.CompetingSamplers
using LinearAlgebra
using LogExpFunctions

# find a low-energy point, set it as samplers state
res = parallel_run(ns, rng, TE=TE);
vmin, imin = findmin(res.trVs[end])
ns.x .= res.xarray[end][imin]
ns.curV[] = vmin

# create a transition matrix for (i,eps) and fixed x=xmin
function buildP(fbdr::FBDRSampler)
    N       = fbdr.np.N
    nlvls   = N+1
    nstates = 2nlvls
    Prows   = [zeros(nstates) for _ in 1:nstates];
    for ieps in 1:2
        # ieps=1
        o = ieps == 1 ? 0 : nlvls
        fbdr.ip[2] = ieps == 1 ? 1 : -1
        for ii in 1:nlvls
            # ii = 1
            pidx = o+ii
            fbdr.ip[1] = ii-1
            NRSTExp.CompetingSamplers.update_ms!(fbdr)
            lpff = log1mexp(min(0., logsumexp(fbdr.ms)))

            # move i
            msidxs = ieps==1 ? ((ii+1):nlvls) : (1:(ii-1))
            jdxs   = o .+ msidxs
            if length(jdxs) > 0
                Prows[pidx][jdxs] .= exp.(fbdr.ms[msidxs])
            end

            # flip eps
            fbdr.ip[2] *= -1                             # simulate flip
            NRSTExp.CompetingSamplers.update_ms!(fbdr)   # recompute IMGS probabilities
            lpfb = log1mexp(min(0., logsumexp(fbdr.ms))) # logprob of failing to sample from {j: (j-i)eps'>0} with the flipped eps'=-eps. also, need to trunc to avoid issues with numerical noise
            Λ    = max(0., exp(lpff) - exp(lpfb))        # exp(a) - exp(b) = exp(b)(exp(a-b)-1)=exp(b)expm1(a-b)
            Prows[pidx][nlvls-o+ii] = Λ
            fbdr.ip[2] *= -1 # undo flip

            # stay
            Prows[pidx][pidx] = max(0., min(1., exp(lpff) - Λ))
        end
    end
    P = collect(hcat(Prows...)')
    return P
end
P = buildP(fbdr)
# diagnostics
all(sum(P,dims=2) .≈ 1.)
T⁺ = P[1:nlvls,1:nlvls]
Λ⁺ = diag(P[1:nlvls,(nlvls+1):2nlvls])
Λ⁻ = diag(P[(nlvls+1):2nlvls, 1:nlvls])
T⁻ = P[(nlvls+1):2nlvls,(nlvls+1):2nlvls]
diag(T⁺) ≈ diag(T⁻)
Λ⁺ - Λ⁻ ≈ sum(T⁻,dims=2) - sum(T⁺,dims=2)

# get stationary distribuion <=> get left-nullspace of P-I 
# <=> get right-nullspace of P'-I
π∞ = nullspace(P'-I)[:,1]
π∞ = π∞ / sum(π∞)
all(π∞ .>= -eps()) # <=1 implicit by imposing sum()=1
π∞[1:nlvls] ≈ π∞[(nlvls+1):2nlvls] # i is indep of eps under stationary dist
2π∞[1:nlvls] ≈ exp.(fbdr.gs) # times 2 <=> marginalize eps


M = collect(hcat(map(idx -> MetroGibbs(fbdr.gs,idx),1:nlvls)...)')
π∞_M = nullspace(M'-I)[:,1]
π∞_M = π∞_M ./ sum(π∞_M)
π∞_M ≈ exp.(fbdr.gs) # correct stationary distribution
diag(T⁻) ≈ (1 .+ diag(M) - (Λ⁺ + Λ⁻))/2 # T^{+}_{ii} = T^{-}_{ii} = [1 + T_{ii} - (Lam_i^{+-}  + Lam_i^{-+}) ]/2

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.9 \
#     gam=8.0  \
#     xps=1e-5 \
#     seed=1111

###############################################################################
# end
###############################################################################
