using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = TitanicHS()
rng = SplittableRandom(753)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    γ=10.,
    maxcor=0.5,
)

using NRSTExp.CompetingSamplers
using LinearAlgebra
using LogExpFunctions




all(sum(P,dims=2) .≈ 1.)

# get stationary distribuion <=> get left-nullspace of P-I 
# <=> get right-nullspace of P'-I
π∞ = nullspace(P'-I)[:,1]
π∞ = π∞ / sum(π∞)
all(π∞ .>= -eps())                 # <=1 implicit by imposing sum()=1
π∞[1:nlvls] ≈ π∞[(nlvls+1):2nlvls] # i is indep of eps under stationary dist
2π∞[1:nlvls] ≈ π∞_tru              # times 2 <=> marginalize eps

function splitTransMat(P::Matrix)
    nlvls = size(P,1)÷2
    T⁺ = P[1:nlvls,1:nlvls]
    T⁻ = P[(nlvls+1):2nlvls,(nlvls+1):2nlvls]
    Λ⁺ = diag(P[1:nlvls,(nlvls+1):2nlvls])
    Λ⁻ = diag(P[(nlvls+1):2nlvls, 1:nlvls])
    return T⁺, T⁻, Λ⁺, Λ⁻
end

# check basic SDBC properties
T⁺, T⁻, Λ⁺, Λ⁻ = splitTransMat(P)
Λ⁺ - Λ⁻ ≈ sum(T⁻ - T⁺,dims=2)      # this is implied
diag(T⁺) ≈ diag(T⁻)                # this condition is not explicit in SDBC papers but it's true

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
