# JULIA_DEBUG=NRST julia --project -t 4 -e "using NRSTExp; dispatch()" exp=benchmark mod=MRNATrans fun=median cor=0.6 gam=2.0 seed=3990

using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

tm  = HalfCauchyEnergy(1000.0)
rng = SplittableRandom(4)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean = false,
    maxcor   = 0.8,
    γ        = 4.0
);



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

