##############################################################################
# simple logistic regression with t(3) priors
#     α   ~ t(3)
#     β_j ~ t(3) iid
#     y_i ~ Bernoulli(logistic(α + <β,x_i>))
##############################################################################

#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Titanic{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    X::Matrix{TF}
    y::BitVector
    Xβ::Vector{TF}
    n::TI
    p::TI
    lenx::TI
    T::TDist{TF}
end
function Titanic()
    X, y = titanic_load_data()
    n, p = size(X)
    Titanic(X, y, similar(X,n), n, p, 1+p, TDist(3))
end

# copy method. keeps everything common except temp storage Xβ. this is needed
# in order to avoid race conditions when sampling in parallel
function Base.copy(tm::Titanic)
    Titanic(tm.X, tm.y, similar(tm.X, tm.n), tm.n, tm.p, tm.lenx, tm.T)
end

# split x into components
function split(tm::Titanic{TF}, x::AbstractVector{TF}) where {TF}
    ( α = x[1], β = view(x, 2:tm.lenx) )
end

# methods for the prior
function NRST.Vref(tm::Titanic{TF}, x) where {TF}
    α, β       = split(tm, x)
    acc        = zero(TF)
    isinf(acc -= logpdf(tm.T, α)) && return acc
    for βᵢ in β
        isinf(acc -= logpdf(tm.T, βᵢ)) && return acc
    end
    return acc
end
Random.rand!(tm::Titanic, rng, x) = rand!(rng, tm.T, x)
function Base.rand(tm::Titanic{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::Titanic{TF}, x) where {TF}
    α, β = split(tm, x)
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
@model function TitanicTuring(X, y)
    N,D = size(X)
    α ~ TDist(3)                        # Intercept
    β = similar(X, D)
    for j in 1:D
        β[j] ~ TDist(3)
    end
    Xβ = X * β
    for i in 1:N
        y[i] ~ BernoulliLogit(α + Xβ[i])
    end
end

# # check that both match
# tm  = Titanic()
# rng = SplittableRandom(44697)
# ns, TE, Λ = NRSTSampler(
#     tm,
#     rng,
#     tune=false
# );
# tmT = NRST.TuringTemperedModel(TitanicTuring(tm.X,tm.y))
# rand!(tm,rng,ns.x)
# NRST.Vref(tm, ns.x) ≈ NRST.Vref(tmT, ns.x)
# NRST.V(tm, ns.x) ≈ NRST.V(tmT, ns.x)
