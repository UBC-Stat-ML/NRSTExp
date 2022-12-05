using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = XYModel(8)# TitanicHS()
rng = SplittableRandom(999)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);


using NRSTExp.CompetingSamplers
using Plots

fbdr = FBDRSampler(ns);
NRST.tour!(fbdr,rng)
ntours = 2048# NRST.min_ntours_TE(TE);
res = parallel_run(fbdr,rng,ntours);

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
#     xps=true \
#     seed=1111

###############################################################################
# end
###############################################################################
