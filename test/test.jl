using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
const tm = Funnel();
const rng = SplittableRandom(1529)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    # nexpl  = 3,
    # maxcor = 0.9,
    # adapt_nexpls = true,
    # γ=3.
);
res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.5  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=MRNATrans  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.5  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=40322

###############################################################################
# end
###############################################################################


