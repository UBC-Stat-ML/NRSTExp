function mixed_lin_log_grid(N::Int)
    N_lin = N÷2
    N_log = N-N_lin
    sort([range(0.,1-inv(N),N_lin); 2. .^ range(-55,0,N_log)])
end

# sample from log-probs
# adapted from StatsBase method: https://github.com/JuliaStats/StatsBase.jl/blob/bd4ca61f4bb75f2c6cd0a47aee1cfde7b696eb9c/src/sampling.jl#L552
# when we have ps = exp.(lps), then sampling does 
#     U ~ U[0,1]
#     n(U) = min{n in 1:m : sum_{j=1}^n ps[j] >= U }
# but 
#     sum_{j=1}^n ps[j] >= U <=> log(sum_{j=1}^n ps[j]) >= log(U)
# Hence, we can sample from lps via
#     E ~ Exp(1)
#     n(E) = min{n in 1:m : logsumexp(ps[1:n]) >= -E }
# Note: can use recursive computation of cumulative logsumexp
#     clp_n = logsumexp(lps[1:n]) = log(exp(lps[n]) + sum_{j=1}^{n-1} exp(lps[j]) )
#           = log(exp(lps[n]) + exp(clp_{n-1}))
#           = logaddexp(clp_{n-1}, lps[n])
# Finally: sum(exp,lps) < 1 is allowed by having a default return value -1,
# which represents that the missing mass was selected.
function sample_logprob(rng::AbstractRNG, lps::Vector{TF}; tol=1000eps(TF)) where {TF<:AbstractFloat}
    nE  = -randexp(rng)
    M   = length(lps)
    m   = 1
    clp = first(lps)
    while clp < nE && m < M
        m += 1
        @inbounds clp = logaddexp(clp, lps[m])
    end
    if clp > tol
        throw(ArgumentError("sample_logprob: excessive mass in lps: logsumexp" *
                            "(lps) = $clp > tol=$tol."))
    elseif clp < nE
        return -1
    else
        return m
    end
end
