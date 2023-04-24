##############################################################################
# simple logistic regression
#     σ   ~ Exp(1)     
#     α   ~ Cauchy(0,σ)
#     β_j ~ Cauchy(0,σ) iid
#     y_i ~ Bernoulli(logistic(α + <β,x_i>))
##############################################################################

#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct TitanicNoQR{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    X::Matrix{TF}
    y::BitVector
    Xβ::Vector{TF}
    n::TI
    p::TI
    lenx::TI
    E::Exponential{TF}
end
function TitanicNoQR()
    X, y = titanic_load_data()
    n, p = size(X)
    TitanicNoQR(X, y, similar(X,n), n, p, 2+p, Exponential())
end

# copy method. keeps everything common except temp storage Xβ. this is needed
# in order to avoid race conditions when sampling in parallel
function Base.copy(tm::TitanicNoQR)
    TitanicNoQR(tm.X, tm.y, similar(tm.X, tm.n), tm.n, tm.p, tm.lenx, tm.E)
end

# split x into components
function split(tm::TitanicNoQR{TF}, x::AbstractVector{TF}) where {TF}
    ( lσ = x[1], α = x[2], β = view(x, 3:tm.lenx) )
end

# methods for the prior
function NRST.Vref(tm::TitanicNoQR{TF}, x) where {TF}
    lσ, α, β   = split(tm, x)
    σ          = exp(lσ)
    (σ < eps(TF)) && return TF(Inf)
    acc        = zero(TF)
    isinf(acc -= logpdf(tm.E, σ)) && return acc
    acc       -= lσ                                        # logabsdetjac
    C          = Cauchy(zero(TF), σ)
    isinf(acc -= logpdf(C, α)) && return acc
    for βᵢ in β
        isinf(acc -= logpdf(C, βᵢ)) && return acc
    end
    return acc
end
function Random.rand!(tm::TitanicNoQR{TF}, rng, x) where {TF}
    σ    = rand(rng, tm.E)
    x[1] = log(σ)
    C    = Cauchy(zero(TF), σ)
    x[2] = rand(rng, C)
    for i in 3:tm.lenx
        x[i] = rand(rng, C)
    end
    return x
end
function Base.rand(tm::TitanicNoQR{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::TitanicNoQR{TF}, x) where {TF}
    _, α, β = split(tm, x)
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
@model function TitanicNoQRTuring(X, y)
    N,D = size(X)
    σ ~ Exponential()
    α ~ Cauchy(0.,σ)                        # Intercept
    β = similar(X, D)
    for j in 1:D
        β[j] ~ Cauchy(0.,σ)
    end
    Xβ = X * β
    for i in 1:N
        y[i] ~ BernoulliLogit(α + Xβ[i])
    end
end

# # check that both match
# tm  = TitanicNoQR()
# rng = SplittableRandom(44697)
# ns, TE, Λ = NRSTSampler(
#     tm,
#     rng,
#     tune=false
# );
# tmT = NRST.TuringTemperedModel(NRSTExp.ExamplesGallery.TitanicNoQRTuring(tm.X,tm.y))
# rand!(tm,rng,ns.x)
# NRST.Vref(tm, ns.x) ≈ NRST.Vref(tmT, ns.x)
# NRST.V(tm, ns.x) ≈ NRST.V(tmT, ns.x)
