###############################################################################
# compare NRST against competitors, using SH16 tuning for competitors
###############################################################################

function benchmark_own_tuning(
    st::NRST.AbstractSTSampler,
    rng::AbstractRNG,
    TE::AbstractFloat,
    Λ::AbstractFloat,
    name::AbstractString
    )
    df = DataFrame()                                      # init empty DataFrame
    benchmark_sampler!(st,rng,df,id=name,TE=TE)
    
    # add extra metadata
    insertcols!(df,
        :N => st.np.N, :Lambda => Λ, :sum_nexpls => sum(st.np.nexpls)
    )
    return df
end
