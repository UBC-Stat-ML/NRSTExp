###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

function benchmark(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat, Λ::AbstractFloat)
    df = DataFrame()                                      # init empty DataFrame
    
    # NRST
    benchmark_sampler!(ns,rng,df,id="NRST",TE=TE)
    ntours_short = max(2_048, ceil(Int, df[1,:ntours]/10)) # to estimate TE in other samplers

    # CompetingSamplers
    ## Simulated Tempering
    ### GT95
    benchmark_sampler!(GT95Sampler(ns),rng,df,id="GT95",ntours_short=ntours_short)

    ### SH16
    benchmark_sampler!(SH16Sampler(ns),rng,df,id="SH16",ntours_short=ntours_short)

    ### FBDR
    benchmark_sampler!(FBDRSampler(ns),rng,df,id="FBDR",ntours_short=ntours_short)

    # # IdealIndexProcesses
    # # BouncyMC: perfect tuning
    # tlens, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
    # TE    = get_TE(vNs)
    # nvevs = -1 # technically infty so it's not meaningful
    # saveres!(df, "DTPerf", tlens, nvevs, TE, ntours)

    # # BouncyMC: actual rejections
    # tlens, vNs = run_tours!(BouncyMC(R), ntours)
    # TE    = get_TE(vNs)
    # nvevs = -1 # technically infty so it's not meaningful
    # saveres!(df, "DTAct", tlens, nvevs, TE, ntours)

    # add extra metadata
    insertcols!(df,
        :N => ns.np.N, :Lambda => Λ, :sum_nexpls => sum(ns.np.nexpls)
    )
    return df
end
