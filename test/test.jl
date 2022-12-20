using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = ChalLogistic()
rng = SplittableRandom(5470)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    N=12,
    adapt_N_rounds=0
)
N     = ns.np.N
nlvls = N+1
atol  = √eps()

using NRSTExp.CompetingSamplers
using LinearAlgebra
using LogExpFunctions

include("testutils.jl")

ri = 9
dc    = (ns.np.c[ri+1]-ns.np.c[ri]) / (ns.np.betas[ri+1]-ns.np.betas[ri])
ns.curV[] = dc

# instantiate samplers
sh   = SH16Sampler(ns);
fbdr = FBDRSampler(ns);
NRSTExp.CompetingSamplers.update_gs!(fbdr);
π∞_tru = exp.(fbdr.gs); # actual target == conditional dist of beta given V

P = buildTransMat(sh)
all(sum(P,dims=2) .≈ 1.)

# get stationary distribuion <=> get left-nullspace of P-I 
# <=> get right-nullspace of P'-I
π∞ = nullspace(P'-I)[:,1]
π∞ = π∞ / sum(π∞)
all(π∞ .>= -eps())                 # <=1 implicit by imposing sum()=1
π∞[1:nlvls] ≈ π∞[(nlvls+1):2nlvls] # i is indep of eps under stationary dist
2π∞[1:nlvls] ≈ exp.(fbdr.gs)       # times 2 <=> marginalize eps

# check basic SDBC properties
T⁺, T⁻, Λ⁺, Λ⁻ = splitTransMat(P)
Λ⁺ - Λ⁻ ≈ sum(T⁻ - T⁺,dims=2)      # this is implied
isapprox(diag(T⁺), diag(T⁻), atol=atol) # this condition is not explicit in SDBC papers but it's true. Need atol due to cases where all entries are ~0

# check formula for staying probs holds
# need to build the transition matrix of the original (reversible) symmetric random walk
M = symRandWalkTransMat(fbdr.gs)
π∞_M = nullspace(M'-I)[:,1]
π∞_M = π∞_M ./ sum(π∞_M)
π∞_M ≈ π∞_tru                      # correct stationary distribution
isapprox(2diag(T⁻), 2diag(M)-(Λ⁺ + Λ⁻), atol=atol) # formula for prob of staying (not explicit in the papers either)

# compute transition matrix for a symmetric random walk from the Gibbs logprobs
function symRandWalkTransMat(gs::AbstractVector)
    nlvls = length(gs)
    Mrows = [zeros(nlvls) for _ in 1:nlvls];
    for ii in 1:nlvls
        # ii = 1
        jj = ii+1
        jj <= nlvls && (Mrows[ii][jj] = 0.5min(1., exp(gs[jj] - gs[ii])))
        jj = ii-1
        jj > 0 && (Mrows[ii][jj] = 0.5min(1., exp(gs[jj] - gs[ii])))
        Mrows[ii][ii] = max(0., min(1., 1. - sum(Mrows[ii])))
    end
    M = collect(hcat(Mrows...)')
    return M
end

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
