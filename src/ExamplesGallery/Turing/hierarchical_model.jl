#######################################
# pure julia version
# >4 times faster than Turing
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct HierarchicalModel{TF<:AbstractFloat,TI<:Int} <: TemperedModel
    τ²_prior::InverseGamma{TF}
    σ²_prior::InverseGamma{TF}
    Y::Matrix{TF}
    N::TI
    J::TI
    lenx::TI
end
function HierarchicalModel()
    Y = hm_load_data()
    HierarchicalModel(InverseGamma(.1,.1), InverseGamma(.1,.1), Y, size(Y)..., 11)
end
function hm_load_data()
    readdlm(pkgdir(NRSTExp, "data", "simulated8schools.csv"), ',', Float64)
end
function invtrans(x::AbstractVector{<:AbstractFloat})
    (τ²=exp(x[1]), σ²=exp(x[2]), μ=x[3], θ = @view x[4:end])
end

# methods for the prior
function NRST.Vref(tm::HierarchicalModel{TF}, x) where {TF}
    τ², σ², μ, θ = invtrans(x)
    acc = zero(TF)
    acc -= logpdf(tm.τ²_prior, τ²) # τ²
    acc -= x[1]                                               # logdetjac τ²
    acc -= logpdf(tm.σ²_prior, σ²) # σ²
    acc -= x[2]                                               # logdetjac σ²
    acc -= logpdf(Cauchy(), μ)                                # μ
    acc -= logpdf(MvNormal(Fill(μ,tm.J), τ²*I), θ)            # θ
    return acc
end
function Random.rand!(tm::HierarchicalModel, rng, x)
    τ²   = rand(rng, tm.τ²_prior)
    τ    = sqrt(τ²)
    x[1] = log(τ²)
    x[2] = log(rand(rng, tm.σ²_prior))
    μ    = rand(rng, Cauchy())
    x[3] = μ
    for i in 4:tm.lenx
        x[i] = rand(rng, Normal(μ, τ))
    end
    return x
end
function Base.rand(tm::HierarchicalModel{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::HierarchicalModel{TF}, x) where {TF}
    _, σ², _, θ = invtrans(x)
    Σ   = σ²*I
    acc = zero(TF)
    for (j, y) in enumerate(eachcol(tm.Y))
        acc -= logpdf(MvNormal(Fill(θ[j], tm.N), Σ), y)
    end
    return acc
end

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
@model function _HierarchicalModelTuring(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                  # can't use improper prior in NRST
    θ  ~ MvNormal(Fill(μ,J), τ²*I)
    for j in 1:J
        Y[:,j] ~ MvNormal(Fill(θ[j], N), σ²*I)
    end
end

# Loading the data and instantiating the model
function HierarchicalModelTuring()
    Y     = readdlm(pkgdir(NRSTExp, "data", "simulated8schools.csv"), ',', Float64)
    model = _HierarchicalModelTuring(Y)
    return TuringTemperedModel(model)
end
