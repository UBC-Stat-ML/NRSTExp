# define the HalfCauchy(0,γ) distribution
struct HalfCauchy{TF<:AbstractFloat} <: Distributions.ContinuousUnivariateDistribution
    C::Cauchy{TF}
end
HalfCauchy(γ=1.0) = HalfCauchy(Cauchy(zero(γ), γ))
function Distributions.logpdf(d::HalfCauchy{TF}, x::Real) where {TF}
    insupport(d, x) ? logtwo + logpdf(d.C, x) : TF(-Inf)
end
Base.rand(rng::AbstractRNG, d::HalfCauchy) = abs(rand(rng, d.C))
Base.minimum(::HalfCauchy{TF}) where {TF} = zero(TF)
Base.maximum(::HalfCauchy{TF}) where {TF} = TF(Inf)

# define the non-standardized Student's t-distribution NSTDist(ν,μ,σ)
struct NSTDist{TF<:AbstractFloat} <: Distributions.ContinuousUnivariateDistribution
    T::TDist{TF}
    μ::TF
    σ::TF
    lσ::TF
end
NSTDist(ν,μ,σ) = NSTDist(TDist(ν), μ, σ, log(σ))
Distributions.logpdf(d::NSTDist, x::Real) = logpdf(d.T, (x-d.μ)/d.σ) - d.lσ
Base.rand(rng::AbstractRNG, d::NSTDist) = d.μ + d.σ*rand(rng, d.T)
Base.minimum(::NSTDist{TF}) where {TF} = TF(-Inf)
Base.maximum(::NSTDist{TF}) where {TF} = TF(Inf)
