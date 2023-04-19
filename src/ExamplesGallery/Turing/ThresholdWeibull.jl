##############################################################################
# Implements the 3-parameter Weibull model and data from Chapter 8 in
#     Cheng, R. (2017). Non-standard parametric statistical inference.
# The model features an unbounded likelihood function
# The full generative model (prior created by me) is
#    a ~ U(0,200)
#    b ~ Inv-Gamma(.1,.1)
#    c ~ U(0.1, 10)
# y|abc~ Weibull_3(y;a,b,c) = 1{y>a}Weibull_2(y-a;b,c)
##############################################################################

#######################################
# pure julia version
#######################################

abstract type AbstractThresholdModel{TF<:AbstractFloat, TI<:Int} <: TemperedModel end

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct ThresholdWeibull{TF, TI} <: AbstractThresholdModel{TF, TI}
    ys::Vector{TF}
    a_prior::Uniform{TF}
    b_prior::InverseGamma{TF}
    c_prior::Uniform{TF}
    lenx::TI
end
function ThresholdWeibull()
    ThresholdWeibull(
        vec(readdlm(pkgdir(NRSTExp, "data", "SteenStickler0.csv"), ',')),
        Uniform(0, 200),
        InverseGamma(.1,.1),
        Uniform(0.1, 10),
        3
    )
end

# methods for the prior
function NRST.Vref(tm::AbstractThresholdModel{TF}, x) where {TF}
    acc  = -logpdf(tm.a_prior, x[1])
    x[2] < zero(TF) && return TF(Inf)
    acc -=  logpdf(tm.b_prior, x[2])
    acc -=  logpdf(tm.c_prior, x[3])
end
function Random.rand!(tm::AbstractThresholdModel, rng, x)
    x[1] = rand(rng, tm.a_prior)
    x[2] = rand(rng, tm.b_prior)
    x[3] = rand(rng, tm.c_prior)
    return x
end
function Base.rand(tm::AbstractThresholdModel{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::ThresholdWeibull{TF}, x) where {TF}
    lik = Weibull(x[3],x[2])
    a   = first(x)
    acc = zero(TF)
    for y in tm.ys
        isinf(acc -= logpdf(lik, y-a)) && break
    end
    return acc
end
