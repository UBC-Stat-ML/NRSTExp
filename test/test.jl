using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

tm  = MRNATrans()
rng = SplittableRandom(1474499973)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);

