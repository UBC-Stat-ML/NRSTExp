###############################################################################
# only run NRST for a given combination of parameters. used for finding good
# hyperparameters
###############################################################################

function hyperparams(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat, Λ::AbstractFloat)
    df = DataFrame()                              # init empty DataFrame
    benchmark_sampler!(ns,rng,df,id="NRST",TE=TE) # NRST
    insertcols!(df,                               # add extra metadata
        :N => ns.np.N, :Lambda => Λ, :sum_nexpls => sum(ns.np.nexpls)
    )
    return df
end
