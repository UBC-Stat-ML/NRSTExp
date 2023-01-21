###############################################################################
# only run NRST for a given combination of parameters. used for finding good
# hyperparameters
###############################################################################

function hyperparams(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE)
    tlens = tourlengths(res)
    nvevs = get_nvevals(res, ns.np.nexpls)
    TE    = last(res.toureff)
    saveres!(df, "NRST", tlens, nvevs, TE, NRST.get_ntours(res))

    # add other metadata
    N = ns.np.N
    Λ = sum(NRST.averej(res))
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)

    return df
end
