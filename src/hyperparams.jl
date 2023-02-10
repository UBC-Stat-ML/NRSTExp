###############################################################################
# only run NRST for a given combination of parameters. used for finding good
# hyperparameters
###############################################################################

function hyperparams(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat, Λ::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE)
    ξ, _  = fit_gpd(res)                 # compute tail index of the distribution of number of visits to the top
    tlens = tourlengths(res)
    nvevs = NRST.get_nvevals.(res.trvec)
    TE    = last(res.toureff)
    nvtop = res.visits[end,1]+res.visits[end,2]
    saveres!(df, "NRST", tlens, nvevs, TE, NRST.get_ntours(res), ξ, nvtop)

    # add other metadata
    N     = ns.np.N
    insertcols!(df,
        :N => N, :Lambda => Λ, :sum_nexpls => sum(ns.np.nexpls)
    )

    return df
end
