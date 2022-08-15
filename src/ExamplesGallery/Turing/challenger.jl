# Define a model using the `DynamicPPL.@model` macro.
@model function _chal_logistic(xs,ys)
    β₀ ~ Normal(0., 10.) # intercept
    β₁ ~ Normal(0., 10.) # slope
    for n in eachindex(ys)
        ys[n] ~ Bernoulli(logistic(β₀ + β₁*xs[n]))
    end
end

# Loading the data and instantiating the model
function ChalLogistic()
    dta   = readdlm(pkgdir(NRSTExp, "data", "challenger.csv"), ',')
    ys    = dta[2:end,1] .> 0
    xs    = Float64.(dta[2:end,2])
    model = _chal_logistic(xs,ys)
    return TuringTemperedModel(model)
end
