using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = HierarchicalModel()# TitanicHS()
rng = SplittableRandom(999)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);

using NRSTExp.CompetingSamplers

gt = GT95Sampler(ns);
ntours = NRST.min_ntours_TE(TE);
res = parallel_run(gt,rng,ntours);
last(res.toureff)

NRST.get_trace(gt).trXplAP