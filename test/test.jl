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

gt = FBDRSampler(ns);
ntours = NRST.min_ntours_TE(TE);
res = parallel_run(gt,rng,ntours);
last(res.toureff)

NRST.get_trace(gt).trXplAP

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.9 \
#     gam=8.0  \
#     seed=1111

###############################################################################
# end
###############################################################################
