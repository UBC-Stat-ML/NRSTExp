#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct ChalLogistic{TF<:AbstractFloat,TMVN<:AbstractMvNormal} <: TemperedModel
    σ₀::TF
    σ₀²::TF
    xs::Vector{TF}
    ys::BitVector
    MVN::TMVN
end
function ChalLogistic()
    σ₀  = 10.
    σ₀² = σ₀*σ₀
    mvn = MvNormal(Diagonal(Fill(σ₀², 2)))
    ChalLogistic(σ₀, σ₀², chal_load_data()..., mvn)
end

function chal_load_data()
    dta = readdlm(pkgdir(NRSTExp, "data", "challenger.csv"), ',', skipstart=1)
    ys  = dta[:,1] .> 0
    xs  = dta[:,2]
    return (xs,ys)
end

# methods for the prior
NRST.Vref(tm::ChalLogistic, x)         = -logpdf(tm.MVN, x)
Random.rand!(tm::ChalLogistic, rng, x) = rand!(rng, tm.MVN, x)
Base.rand(tm::ChalLogistic, rng)       = rand(rng, tm.MVN)

# method for the likelihood potential
# y|x,β ~ Bern(p(xβ))
# L(β) = prod_{i=1}^n p(x_iβ)^{y_i}(1-p(x_iβ))^{1-y_i}
# ℓ(β) = sum_{i=1}^n y_i log[p(x_iβ)] + (1-y_i)log[(1-p(x_iβ))]
# p(xβ) = [1+e^{-xβ}]^{-1}
# => log[p(xβ)] = -log[1+exp(-xβ)] = -log1pexp(-xβ) ✓
# 1-p(xβ) = [1+e^{-xβ}-1]/[1+e^{-xβ}] = e^{-xβ}/[1+e^{-xβ}] = 1/[1+e^{xβ}]
# => log[1-p(xβ)] = -log[1+exp(xβ)] = -log1pexp(xβ) ✓
function NRST.V(tm::ChalLogistic{TF}, β) where {TF}
    β₀ = β[1]; β₁ = β[2]
    vacc = zero(TF)
    for (i,y) in enumerate(tm.ys)
        @inbounds xβ = β₀ + β₁*tm.xs[i]
        vacc += y ? log1pexp(-xβ) : log1pexp(xβ)
    end
    return vacc
end

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
@model function _chal_logistic(xs,ys)
    β₀ ~ Normal(0., 10.) # intercept
    β₁ ~ Normal(0., 10.) # slope
    for n in eachindex(ys)
        ys[n] ~ BernoulliLogit(β₀ + β₁*xs[n])
    end
end

# Loading the data and instantiating the model
function ChalLogisticTuring()
    xs, ys = chal_load_data()
    model  = _chal_logistic(xs,ys)
    return TuringTemperedModel(model)
end
