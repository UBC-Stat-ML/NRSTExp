##############################################################################
# Neal's funnel in arbitrary dimensions
# using the parameters from https://arxiv.org/abs/2110.00610
#     β ~ N(0,σ^2)
#  αᵢ|β ~ N(0,e^β) (iid)
# Need to build a reference and corresponding V function
# Reference:
#    βᵢ ~ N(0,σ^2)
#    αᵢ ~ N(0,σ₂^2) (iid)
# then
#    pi_1(x) = pi_0 (pi_1/pi_0) = pi_0 prod_{i=2}^d N(xi; 0,e^β)/N(xi; σ₂^2)
#            = pi_0 exp[-sum_{i=2}^d logpdf(N(xi; σ₂^2)) - logpdf(N(xi; 0,e^β))]
# so
#    V(x) := sum_{i=2}^d logpdf(N(xi; σ₂^2)) - logpdf(N(xi; 0,e^β))
##############################################################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Funnel{TF<:AbstractFloat,TI<:Int} <: TemperedModel
    β_dist::Normal{TF}
    α_ref::Normal{TF}
    lenx::TI
end
function Funnel(;σ=3., σ₂=3., d=20)
    Funnel(Normal(zero(σ), σ), Normal(zero(σ), σ₂), d)
end

# methods for the reference
function NRST.Vref(tm::Funnel, x)
    @inbounds -(logpdf(tm.β_dist, x[1]) + sum(xi -> logpdf(tm.α_ref, xi), x[2:end]))
end
function Random.rand!(tm::Funnel{TF}, rng, x) where {TF}
    x[begin] = rand(rng, tm.β_dist)
    for i in 2:tm.lenx
        @inbounds x[i] = rand(rng, tm.α_ref)
    end
    return x
end
function Base.rand(tm::Funnel{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the potential
function NRST.V(tm::Funnel{TF}, x) where {TF}
    α_dist = Normal(zero(TF), exp(first(x)))
    acc    = zero(TF)
    @inbounds for i in 2:tm.lenx
        isinf(acc += logpdf(tm.α_ref, x[i]) - logpdf(α_dist, x[i])) && break
    end
    return acc
end

# using Printf, Plots, ColorSchemes
# res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);
# X = collect(hcat(res.xarray[end]...)');
# colorgrad = cgrad([ColorSchemes.viridis[begin], ColorSchemes.viridis[end]],ns.np.N+1 )
# anim = @animate for (i,xs) in enumerate(res.xarray)
#     β = @sprintf("%.2e", ns.np.betas[i])
#     X = collect(hcat(xs...)');
#     plt = scatter(
#         X[:,begin], X[:,end], title = "β = $β", label="",
#         xlabel="x[1]", ylabel="x[20]", xlims=(-10,10), ylims=(-50,50),
#         markercolor = colorgrad[i], 
#     )
# end
# gif(anim, "funnel.gif", fps=2)