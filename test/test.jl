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
