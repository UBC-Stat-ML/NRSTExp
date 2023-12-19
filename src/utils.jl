###############################################################################
# utils
###############################################################################

# compute Tour Effectiveness in one pass
function get_TE(vs::AbstractVector)
    m,s = mean_and_std(vs; corrected=false) # use 1/n instead of 1/(n-1)
    inv(one(m) + abs2(s/m)) # E[v]^2 / E[v^2] = E[v]^2 / [E[v]^2 + var[v]] = 1/( 1 + (sd[v]/E[v])^2 )
end

# common interface for benchmark and hyperparams
function benchmark_sampler!(
    st::NRST.AbstractSTSampler,
    rng::AbstractRNG,
    df::AbstractDataFrame;
    id::AbstractString,
    TE::AbstractFloat = NaN,
    ntours_short::Int = -1,
    α::AbstractFloat  = NRST.DEFAULT_α,
    δ::AbstractFloat  = NRST.DEFAULT_δ
    )
    if isnan(TE) # need to estimate it with a short run
        TE = last(parallel_run(st, rng, ntours=ntours_short).toureff)
    end
    if TE >= NRST.DEFAULT_TE_min # bail if TE is too low
        ntours= NRST.min_ntours_TE(TE,α,δ)
        res   = parallel_run(st, rng, ntours=ntours)
        ξ, _  = fit_gpd(res)                           # compute tail index of the distribution of number of visits to the top
        tlens = tourlengths(res)
        nvevs = NRST.get_nvevals.(res.trvec)
        TE    = last(res.toureff)                      # get better estimate
        nvtop = res.visits[end,1]+res.visits[end,2]
        saveres!(df, id, tlens, nvevs, TE, ntours, ξ, nvtop)
    end
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

