using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

tm  = TitanicHS()
rng = SplittableRandom(1474499973)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    γ=1.0,
    maxcor=0.9
);
