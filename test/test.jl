using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic();
rng = SplittableRandom(2798)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean   = true,
    maxcor     = 0.9,
    γ          = 8.0,
    xpl_smooth_λ = 0.1
)
res=parallel_run(ns,rng,TE=TE);

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=HierarchicalModel  \
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=2798

###############################################################################
# end
###############################################################################
