#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct TitanicHS{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    X::Matrix{TF}
    y::BitVector
    Xβ::Vector{TF}
    n::TI
    p::TI
    lenx::TI
    H::HalfCauchy{TF}
    T::TDist{TF}
end
function TitanicHS()
    X, y = titanic_load_data()
    n, p = size(X)
    TitanicHS(X, y, similar(X,n), n, p, 2 + 2p, HalfCauchy(), TDist(3))
end

# copy method. keeps everything common except temp storage Xβ. this is needed
# in order to avoid race conditions when sampling in parallel
function Base.copy(tm::TitanicHS)
    TitanicHS(tm.X, tm.y, similar(tm.X, tm.n), tm.n, tm.p, tm.lenx, tm.H, tm.T)
end

function invtrans(tm::TitanicHS{TF}, x::AbstractVector{TF}) where {TF}
    p = tm.p
    ( τ = x[1], α = x[2], λ = view(x, 3:(p+2)), β = view(x, (p+3):tm.lenx) )
end

# methods for the prior
function NRST.Vref(tm::TitanicHS{TF}, x) where {TF}
    τ, α, λ, β = invtrans(tm, x)
    acc  = zero(TF)
    isinf(acc -= logpdf(tm.H, τ)) && return acc
    isinf(acc -= logpdf(tm.T, α)) && return acc
    for (i, λᵢ) in enumerate(λ)
        isinf(acc -= logpdf(tm.H, λᵢ)) && return acc
        isinf(acc -= logpdf(Normal(zero(TF), λᵢ*τ), β[i])) && return acc
    end
    return acc
end
function Random.rand!(tm::TitanicHS{TF}, rng, x) where {TF}
    τ    = rand(rng, tm.H)
    x[1] = τ
    x[2] = rand(rng, tm.T)
    p    = tm.p
    for i in 3:(p+2)
        λᵢ     = rand(rng, tm.H)
        x[i]   = λᵢ
        x[i+p] = rand(rng, Normal(zero(TF), λᵢ*τ))
    end
    return x
end
function Base.rand(tm::TitanicHS{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::TitanicHS{TF}, x) where {TF}
    _, α, _, β = invtrans(tm, x)
    mul!(tm.Xβ, tm.X, β)
    acc = zero(TF)
    for (i, yᵢ) in enumerate(tm.y)
        @inbounds ℓ = α + tm.Xβ[i]             # 2.5 times faster with @inbounds
        acc += yᵢ ? log1pexp(-ℓ) : log1pexp(ℓ) # log1pexp is never Inf if ℓ isn't, so no need for a check here
    end
    return acc
end

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
@model function _TitanicHSTuring(X, y)
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
TitanicHSTuring() = TuringTemperedModel(_TitanicHSTuring(titanic_load_data()...))

# # check both give the same
# tm  = TitanicHS()
# rng = SplittableRandom(4)
# ns, TE, Λ = NRSTSampler(
#     tm,
#     rng,
#     tune = false
# );
# tmT = TitanicHSTuring();
# T(x) = vcat(log(x[1]), log.(view(x, 3:(tm.p+2))), x[2], view(x, (tm.p+3):tm.lenx) ) # note: logs and different order
# Tx = T(ns.x)
# NRST.V(tmT, Tx) ≈ NRST.V(tm, ns.x)
# NRST.Vref(tmT,Tx) ≈ NRST.Vref(tm,ns.x) - sum(view(Tx, 1:(tm.p+1))) # need to add -logabsdetjac
