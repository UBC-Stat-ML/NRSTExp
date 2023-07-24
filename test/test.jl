###############################################################################
# IMPORTANT: the current formula for determining ntours does not apply to 
# the median strategy. It is easy to see that it fails since this strategy
# produces lower Lambda, which would appear as a higher TEinfty compared to 
# the mean strategy. However, we know that the median strategy produces lower TEs! 
# TODO: increase by p_ratio in this setting
###############################################################################

###############################################################################
# TODO: use stepping stone 
# 1) forward only when E^{0}[|V|]=infty, E^{N}[|V|]<infty 
# 2) bwd only when E^{N}[|V|]=infty, E^{0}[|V|]<infty 
###############################################################################

using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler
tm = ThresholdWeibull();#HierarchicalModel();#MRNATrans();#ChalLogistic();#MvNormalTM(3,2.,2.)#Funnel();#Banana();#XYModel(8);#MvNormalTM(32,4.,2.);
rng = SplittableRandom(57);
# NRST.V(tm, rand(tm,rng))
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    # max_rounds=18
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


