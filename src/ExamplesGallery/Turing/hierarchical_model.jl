# Define a model using the `DynamicPPL.@model` macro.
@model function HierarchicalModel(Y)
    N,J= size(Y)
    τ² ~ InverseGamma(.1,.1)
    σ² ~ InverseGamma(.1,.1)
    μ  ~ Cauchy()                  # can't use improper prior in NRST
    θ  ~ MvNormal(fill(μ,J), τ²*I)
    for j in 1:J
        Y[:,j] ~ MvNormal(fill(θ[j], N), σ²*I)
    end
end
dataY = readdlm(pkgdir(NRSTExp, "data", "simulated8schools.csv"), ',', Float64)
const HierarchicalModelWithData = HierarchicalModel(dataY)
