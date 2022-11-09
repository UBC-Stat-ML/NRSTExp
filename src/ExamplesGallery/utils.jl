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