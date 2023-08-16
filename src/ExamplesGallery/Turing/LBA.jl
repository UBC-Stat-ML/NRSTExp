# Source: https://github.com/Julia-Tempering/Pigeons.jl/issues/106
# Specify LBA model
@model function model_lba(data; min_rt=minimum(data.rt))
    # Priors
    ν ~ MvNormal(zeros(2), I * 2)
    A ~ truncated(Normal(0.8, 0.4), 0.0, Inf)
    k ~ truncated(Normal(0.2, 0.2), 0.0, Inf)
    τ ~ Uniform(0.0, min_rt)

    # Likelihood
    data ~ LBA(; ν, A, k, τ)
end

function LBAModel(;seed=45461)
    Random.seed!(seed)

    # Generate some data with known parameters
    dist = LBA(ν=[3.0, 2.0], A=0.8, k=0.2, τ=0.3)
    data = rand(dist, 100)

    NRST.TuringTemperedModel(model_lba(data))
end
