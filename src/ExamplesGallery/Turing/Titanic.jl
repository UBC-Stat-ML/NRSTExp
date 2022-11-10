#######################################
# pure julia version
#######################################

function titanic_load_data()
    dta = readdlm(pkgdir(NRSTExp, "data", "titanic.csv"), ',', skipstart=1)
    y   = dta[:,1] .> 0
    X   = dta[:,2:end]
    return X, y
end

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
# followed advices from these two posts
# https://discourse.julialang.org/t/regularized-horseshoe-prior/71599/2?
# https://discourse.julialang.org/t/case-study-speeding-up-a-logistic-regression-with-rhs-prior-turing-vs-numpyro-any-tricks-im-missing/87681/2
@model function _TitanicTuring(X, y)
    # X: matrix N (passengers) times D (predictors)
    # y: BitVector length n
    D = size(X, 2)
    τ ~ HalfCauchy()
    λ ~ filldist(HalfCauchy(), D)
    α ~ TDist(3)                        # Intercept
    β ~ MvNormal(Diagonal((λ .* τ).^2)) # Coefficients
    y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
end

# Loading the data and instantiating the model
function TitanicTuring()
    X, y  = titanic_load_data()
    model = _TitanicTuring(X,y)
    return TuringTemperedModel(model)
end
