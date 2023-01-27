###############################################################################
# utils
###############################################################################

TE_est(vNs::AbstractVector) = (sum(vNs) ^ 2) / (length(vNs)*sum(abs2, vNs))

# compute number of V evaluations per tour. assume 1 per explorer step
function get_nvevals(res::NRST.RunResults, nexpls::AbstractVector)
    map(tr -> get_nvevals(tr, nexpls), res.trvec)
end
function get_nvevals(tr::NRST.AbstractTrace{T,TI}, nexpls::AbstractVector) where {T,TI<:Int}
    sum(ip -> ip[1]>zero(TI) ? nexpls[ip[1]] : one(TI), tr.trIP)
end

# store results into df
function saveres!(df::AbstractDataFrame, proc, tlens, nvevs, TE, ntours)
    append!(df,
        DataFrame(
            proc=proc, rtser=sum(tlens), rtpar=maximum(tlens),
            costser=sum(nvevs), costpar=maximum(nvevs), TE=TE,
            ntours=ntours
        )
    )
end

# fit a GPD to the number of visits to the top level
function fit_gpd(res::NRST.TouringRunResults)
    N     = NRST.get_N(res)
    nvtop = [sum(ip -> first(ip)==N, tr.trIP) for tr in res.trvec]
    @assert sum(nvtop) == (res.visits[end,1]+res.visits[end,2])
    sort!(nvtop)
    idx   = findfirst(x->x>0,nvtop)
    (isnothing(idx) || idx==length(nvtop)) && return (NaN, NaN)
    ParetoSmooth.gpd_fit(nvtop[idx:end] .+ 1e-7, 1.0)              # implicit convert to float by adding small ϵ, which fixes weird behavior in gdp_fit when int-like floats are used. 
end

