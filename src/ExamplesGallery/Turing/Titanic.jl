#######################################
# pure julia version
#######################################

function titanic_load_data()
    dta = readdlm(pkgdir(NRSTExp, "data", "titanic.csv"), ',', skipstart=1)
    y   = dta[:,1] .> 0
    X   = [copy(r) for r in eachrow(dta[:,2:end])]
    return X,y
end

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
@model function _TitanicTuring(X,y)
    # X: vector length N (passengers) of vectors length D (predictors)
    # y: BitVector length n
    D = length(first(X))
    τ ~ HalfCauchy()
    λ ~ filldist(HalfCauchy(), D)
    α ~ TDist(3)                        # Intercept
    β ~ MvNormal(Diagonal((λ .* τ).^2)) # Coefficients
    for (n, x) in enumerate(X)
        ℓ    = α + dot(β,x)
        y[n] ~ BernoulliLogit(ℓ)
    end
end

# Loading the data and instantiating the model
function TitanicTuring()
    X, y  = titanic_load_data()
    model = _TitanicTuring(X,y)
    return TuringTemperedModel(model)
end
