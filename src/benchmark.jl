###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

function benchmark(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE)
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = last(res.toureff)
    ntours= NRST.get_ntours(res)
    saveres!(df, "NRST", tlens, nvevs, TE, ntours)

    # inputs used for other processes
    N = ns.np.N
    R = NRST.rejrates(res)
    Λ = sum(NRST.averej(R))
    ntours_small = max(2048, ceil(Int, ntours/10)) # to estimate TE in other samplers

    # CompetingSamplers
    ## Simulated Tempering
    ### GT95
    gt    = GT95Sampler(ns)
    TE    = last(parallel_run(gt, rng, ntours=ntours_small).toureff)
    res   = parallel_run(gt, rng, TE=TE)
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = last(res.toureff)
    saveres!(df, "GT95", tlens, nvevs, TE, NRST.get_ntours(res))

    ### SH16
    sh    = SH16Sampler(ns)
    TE    = last(parallel_run(sh, rng, ntours=ntours_small).toureff)
    res   = parallel_run(sh, rng, TE=TE)
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = last(res.toureff)
    saveres!(df, "SH16", tlens, nvevs, TE, NRST.get_ntours(res))

    ### FBDR
    fbdr  = FBDRSampler(ns)
    TE    = last(parallel_run(fbdr, rng, ntours=ntours_small).toureff)
    res   = parallel_run(fbdr, rng, TE=TE)    
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = last(res.toureff)
    saveres!(df, "FBDR", tlens, nvevs, TE, NRST.get_ntours(res))

    # IdealIndexProcesses
    # BouncyMC: perfect tuning
    tlens, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so it's not meaningful
    saveres!(df, "DTPerf", tlens, nvevs, TE, ntours)

    # BouncyMC: actual rejections
    tlens, vNs = run_tours!(BouncyMC(R), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so it's not meaningful
    saveres!(df, "DTAct", tlens, nvevs, TE, ntours)


    # add other metadata
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)

    return df
end
