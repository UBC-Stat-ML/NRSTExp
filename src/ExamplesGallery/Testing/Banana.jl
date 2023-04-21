##############################################################################
# Classic Rosenbrock's Banana, as described by eq 1 in
#     https://doi.org/10.1111/sjos.12532
# Generative model
#     x1    ~ N(1,sqrt{10})
#     x2|x1 ~ N(x1^2,1/sqrt{200})  //  2s^2 = 1/100 <=> s^2 = 1/200 -> s = 1/sqrt{200}
# Reference
#     x1,x2 ~ N(1,sqrt{10}) (iid)
# Potential
#     pi(x) = pi0(x) pi(x)/pi0(x)
#           = pi0(x) N(x2; x1^2,1/sqrt{200})/N(x2;1,sqrt{10})
# => V(x) := logpdf(N(x2;1,sqrt{10})) - logpdf(N(x2; x1^2,1/sqrt{200}))
##############################################################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Banana{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    x1_dist::Normal{TF}
    σ_x2::TF
    lenx::TI
end
Banana() = Banana(Normal(1., sqrt(10)), inv(sqrt(200.)), 2)

# methods for the reference
NRST.Vref(tm::Banana, x) = -sum(xi -> logpdf(tm.x1_dist, xi), x)
function Random.rand!(tm::Banana{TF}, rng, x) where {TF}
    for i in eachindex(x)
        @inbounds x[i] = rand(rng, tm.x1_dist)
    end
    return x
end
function Base.rand(tm::Banana{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the potential
function NRST.V(tm::Banana{TF}, x) where {TF}
    logpdf(tm.x1_dist, last(x)) - logpdf(Normal(abs2(first(x)), tm.σ_x2), last(x))
end

# using Plots
# using Printf

# anim = @animate for (i,xs) in enumerate(res.xarray)
#     β = @sprintf("%.2e", ns.np.betas[i])
#     X = collect(hcat(xs...)');
#     plt = scatter(
#         X[:,begin], X[:,end], title = "β = $β", label="",xlabel="x[1]", 
#         ylabel="x[2]", xlims=(-10,10), ylims=(-50,50)
#     )
# end
# gif(anim, "banana.gif", fps=2)