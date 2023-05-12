###############################################################################
# compare NRST against competitors, using SH16 tuning for competitors
###############################################################################

function benchmark_SH16_tuning(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat, Λ::AbstractFloat)
    df = DataFrame()                                      # init empty DataFrame
    
    # NRST
    benchmark_sampler!(ns,rng,df,id="NRST",TE=TE)
    ntours_short = max(2_048, ceil(Int, df[1,:ntours]/10)) # to estimate TE in other samplers

    # CompetingSamplers
    ## Simulated Tempering
    ### SH16
    sh = NRST.init_sampler(SH16Sampler, ns.np.tm, rng, N=512-1) # 512 total grid points
    NRST.tune!(sh,rng)
    benchmark_sampler!(sh,rng,df,id="SH16",ntours_short=ntours_short)

    ### GT95
    benchmark_sampler!(GT95Sampler(sh),rng,df,id="GT95",ntours_short=ntours_short)

    ### FBDR
    benchmark_sampler!(FBDRSampler(sh),rng,df,id="FBDR",ntours_short=ntours_short)

    # add extra metadata
    insertcols!(df,
        :N => ns.np.N, :Lambda => Λ, :sum_nexpls => sum(ns.np.nexpls)
    )
    return df
end
