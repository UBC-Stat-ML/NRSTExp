###############################################################################
# utils
###############################################################################

get_TE(vNs::AbstractVector) = (sum(vNs) ^ 2) / (length(vNs)*sum(abs2, vNs))

# 
function benchmark_sampler!(
    st::NRST.AbstractSTSampler,
    rng::AbstractRNG,
    df::AbstractDataFrame;
    id::AbstractString,
    TE::AbstractFloat = NaN,
    ntours_small::Int = -1,
    α::AbstractFloat  = NRST.DEFAULT_α,
    δ::AbstractFloat  = NRST.DEFAULT_δ
    )
    if isnan(TE)
        TE = last(parallel_run(st, rng, ntours=ntours_small).toureff)
    end
    ntours= NRST.min_ntours_TE(TE,α,δ)
    res   = parallel_run(st, rng, ntours=ntours)
    ξ, _  = fit_gpd(res)                           # compute tail index of the distribution of number of visits to the top
    tlens = tourlengths(res)
    nvevs = NRST.get_nvevals.(res.trvec)
    TE    = last(res.toureff)                      # get better estimate
    nvtop = res.visits[end,1]+res.visits[end,2]
    saveres!(df, id, tlens, nvevs, TE, ntours, ξ, nvtop)
end

# store results into df
function saveres!(df::AbstractDataFrame, proc, tlens, nvevs, TE, ntours, ξ, nvtop)
    append!(df,
        DataFrame(
            proc=proc, rtser=sum(tlens), rtpar=maximum(tlens),
            costser=sum(nvevs), costpar=maximum(nvevs), TE=TE,
            ntours=ntours, xi=ξ, n_vis_top=nvtop
        )
    )
end

# fit a GPD to the number of visits to the top level
function fit_gpd(res::NRST.TouringRunResults)
    nvtop = NRST.get_nvtop.(res.trvec)
    @assert sum(nvtop) == (res.visits[end,1]+res.visits[end,2])
    sort!(nvtop)
    idx   = findfirst(x->x>0,nvtop)
    (isnothing(idx) || idx==length(nvtop)) && return (NaN, NaN)
    ParetoSmooth.gpd_fit(nvtop[idx:end] .+ 1e-7, 1.0)              # implicit convert to float by adding small ϵ, which fixes weird behavior in gdp_fit when int-like floats are used. 
end

