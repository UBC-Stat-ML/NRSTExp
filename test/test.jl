using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

tm  = MRNATrans()
rng = SplittableRandom(868)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);

