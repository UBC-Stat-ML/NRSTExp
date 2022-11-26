using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = TitanicHS()
rng = SplittableRandom(1474499973)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
