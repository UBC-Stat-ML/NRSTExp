###############################################################################
# only run NRST for a given combination of parameters. used for finding good
# hyperparameters
###############################################################################

function hyperparams(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = res.toureff[end]
    saveres!(df, "NRST", tlens, nvevs, TE, NRST.get_ntours(res))

    # add other metadata
    N = ns.np.N
    Λ = sum(NRST.averej(res))
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)

    return df
end
