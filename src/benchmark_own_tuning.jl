###############################################################################
# compare NRST against competitors, using SH16 tuning for competitors
###############################################################################

function benchmark_own_tuning(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat, Λ::AbstractFloat)
    df = DataFrame()                                      # init empty DataFrame
    
    # NRST
    benchmark_sampler!(ns,rng,df,id="NRST",TE=TE)
    ntours_short = max(2_048, ceil(Int, df[1,:ntours]/10)) # to estimate TE in other samplers

    # CompetingSamplers
    ## Simulated Tempering
    ### SH16
    sh = NRST.init_sampler(SH16Sampler, ns.np.tm, rng, N=512-1) # bottleneck is HierarchicalModel
    NRST.tune!(sh,rng)
    t = @async benchmark_sampler!(sh,rng,df,id="SH16",ntours_short=ntours_short)
    status = timedwait(3600) do
        istaskdone(t)
    end
    println("SH16Sampler benchmark status: $status.")

    ### FBDR
    fbdr = NRST.init_sampler(FBDRSampler, ns.np.tm, rng, N=512-1) # bottleneck for N is Funnel
    NRST.tune!(fbdr,rng)
    t = @async benchmark_sampler!(fbdr,rng,df,id="FBDR",ntours_short=ntours_short)
    status = timedwait(3600) do
        istaskdone(t)
    end
    println("FBDRSampler benchmark status: $status.")

    ### GT95
    gt = GT95Sampler(fbdr) # just copy FBDR's config, more reliable than SH16
    t = @async benchmark_sampler!(gt,rng,df,id="GT95",ntours_short=ntours_short)
    status = timedwait(3600) do
        istaskdone(t)
    end
    println("GT95Sampler benchmark status: $status.")

    # add extra metadata
    insertcols!(df,
        :N => ns.np.N, :Lambda => Λ, :sum_nexpls => sum(ns.np.nexpls)
    )
    return df
end
