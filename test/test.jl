# JULIA_DEBUG=NRST julia --project -t 4 -e "using NRSTExp; dispatch()" exp=benchmark mod=MRNATrans fun=mean cor=0.7 gam=5.0 seed=8371

using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

tm  = ChalLogistic()
rng = SplittableRandom(4)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean = true,
    maxcor   = 0.6,
    γ        = 4.0
);

rand()
using Random
randn!(rng, zeros(3))
using Distributions
rand!(Exponential(), zeros(5))
randn!
rand!(rng, HalfCauchy(),zeros(4))
using Distributions, DynamicPPL
using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

# Define a model using the `DynamicPPL.@model` macro
@model function Lnmodel(x)
    s  ~ HalfCauchy()
    x .~ Normal(0.,s)
end

# Now we instantiate a proper `DynamicPPL.Model` object by a passing a vector of observations
model   = Lnmodel(randn(30))
rng     = SplittableRandom(4)
ns, _, _= NRSTSampler(model, rng, γ=3.0);

