###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

function benchmark(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = res.toureff[end]
    saveres!(df, "NRST", tlens, nvevs, TE)

    # TODO: ADD COMPETING SAMPLERS

    # inputs used for ideal processes
    N = ns.np.N
    R = NRST.rejrates(res)
    Λ = sum(NRST.averej(R))
    ntours = NRST.get_ntours(res)

    # BouncyMC: perfect tuning
    tlens, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so it's not meaningful
    saveres!(df, "DTPerf", tlens, nvevs, TE)

    # BouncyMC: actual rejections
    tlens, vNs = run_tours!(BouncyMC(R), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so it's not meaningful
    saveres!(df, "DTAct", tlens, nvevs, TE)

    # add other metadata
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)
    insertcols!(df, :ntours => ntours)

    return df
end
