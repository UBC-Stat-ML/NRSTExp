using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm = ChalLogistic();#Funnel();#ThresholdWeibull();#HierarchicalModel();#Banana();#XYModel(8);#MRNATrans();#MvNormalTM(32,4.,2.);
rng = SplittableRandom(89440)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=TitanicHS  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchOwnTune  \
#     sam=GT95 \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.0  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=89440

###############################################################################
# end
###############################################################################


