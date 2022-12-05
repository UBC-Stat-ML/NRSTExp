using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = TitanicHS()
rng = SplittableRandom(999)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    max_ar_ratio=0.8,
    max_rounds=10
);
ns.np.xplpars
using NRSTExp.CompetingSamplers
using Plots

fbdr = FBDRSampler(ns);
NRST.tour!(fbdr,rng)
ntours = 2048# NRST.min_ntours_TE(TE);
res = parallel_run(fbdr,rng,ntours=ntours);

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
