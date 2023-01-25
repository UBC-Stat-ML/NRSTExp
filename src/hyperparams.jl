###############################################################################
# only run NRST for a given combination of parameters. used for finding good
# hyperparameters
###############################################################################

function hyperparams(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE)
    ξ, _  = fit_gpd(res) # check square-integrability of number of visits to the top
    if isnan(ξ)
        return insertcols!(df,:error => "hyperparams: too few tours with visits to top level.")
    elseif ξ >= 0.5
        return insertcols!(df,:error => "hyperparams: configuration has ξ=$(round(ξ,digits=2))>=0.5 ⟹ Actual TE=0 (estimated TE=$TE).")
    end
    
    # get stats
    tlens = tourlengths(res)
    nvevs = get_nvevals(res, ns.np.nexpls)
    TE    = last(res.toureff)
    saveres!(df, "NRST", tlens, nvevs, TE, NRST.get_ntours(res))

    # add other metadata
    N = ns.np.N
    Λ = sum(NRST.averej(res))
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)
    insertcols!(df, :xi => ξ)
    insertcols!(df, :sum_nexpls => sum(ns.np.nexpls))

    return df
end
