# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct HalfCauchyEnergy{TF<:AbstractFloat} <: TemperedModel
    HC::HalfCauchy{TF}
end
HalfCauchyEnergy(γ::AbstractFloat)   = HalfCauchyEnergy(HalfCauchy(γ))
NRST.V(::HalfCauchyEnergy, v)        = v[1]
NRST.Vref(tm::HalfCauchyEnergy, v)   = -logpdf(tm.HC, v[1])
Base.rand(tm::HalfCauchyEnergy, rng) = [rand(rng, tm.HC)]

# numerical checks
# using QuadGK

# # density of half-Cauchy with scale γ>0
# dhcauchy(x,γ) = 2inv(pi * γ * ( 1 + abs2(x/γ) ))
# 1. ≈ first(quadgk(x->dhcauchy(x,10.), 0., Inf)) # check int () = 1

# # unnormalized density
# q(x,γ,β) = dhcauchy(x,γ)*exp(-β*x)

# function get_skewness(γ,β)
#     c,_  = quadgk(x->q(x,γ,β), 0., Inf)           # calculate normalizing const
#     p(x) = q(x,γ,β)/c                             # define proper pdf
#     @assert 1. ≈ first(quadgk(p, 0., Inf))        # check int () = 1
#     m,_  = quadgk(x->x*p(x), 0., Inf)             # get the mean
#     s²,_ = quadgk(x->abs2(x-m)*p(x), 0., Inf)     # get the variance
#     s    = sqrt(s²)
#     first(quadgk(x->((x-m)/s)^3 * p(x), 0., Inf)) # return skewness
# end
# prodpars = Base.product(2. .^range(-7.5,0,20), 0.05:.05:1)
# skns = [get_skewness(tup...) for tup in prodpars]

# using Plots
# heatmap(
#     prodpars.iterators[1],
#     prodpars.iterators[2],
#     skns,
#     xscale=:log10,
#     xlabel="γ",
#     ylabel="β",
#     title="Skewness as function of (γ,β)"
# )