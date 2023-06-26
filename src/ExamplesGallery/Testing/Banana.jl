##############################################################################
# Classic Rosenbrock's Banana, as described by eq 1 in https://doi.org/10.1111/sjos.12532
#     p(x) \propto exp(-[ (x1-1)^2/20 + (x2-x1^2)/(1/5) ])
# Generative model
#     x1    ~ N(1,sqrt{10})       // using Distributions notation N(mu,sigma)
#     x2|x1 ~ N(x1^2,1/sqrt{10})  // 2s^2 = 1/5 <=> s^2 = 1/10 -> s = 1/sqrt{10}
# Reference
#     x1    ~ N(1,sqrt{10})
#     x2    ~ N(11, 10)           // E[x1^2] = 1^2 + 10 = 11, P(x2 ∈ [1,21]) ~ 68% 
# Potential
#     pi(x) = pi0(x) pi(x)/pi0(x)
#           = pi0(x) N(x2; x1^2,1/sqrt{10})/N(x2;11, 10)
# => V(x) := logpdf(N(x2; 11, 10)) - logpdf(N(x2; x1^2, 1/sqrt{10}))
##############################################################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Banana{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    x1_dist::Normal{TF}
    x2_ref::Normal{TF}
    σ_x2::TF
    lenx::TI
end
Banana() = Banana(Normal(1., sqrt(10)), Normal(11., 10), inv(sqrt(200.)), 2)

# methods for the reference
NRST.Vref(tm::Banana, x) = -(logpdf(tm.x1_dist, first(x))+logpdf(tm.x2_ref, last(x)))
function Random.rand!(tm::Banana, rng, x)
    x[begin] = rand(rng, tm.x1_dist)
    x[end]   = rand(rng, tm.x2_ref)
    return x
end
function Base.rand(tm::Banana{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the potential
function NRST.V(tm::Banana, x)
    logpdf(tm.x2_ref, last(x)) - logpdf(Normal(abs2(first(x)), tm.σ_x2), last(x))
end
