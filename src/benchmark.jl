###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

TE_est(vNs::AbstractVector) = (sum(vNs) ^ 2) / (length(vNs)*sum(abs2, vNs))

function benchmark(ns::NRSTSampler, rng::AbstractRNG, TE::AbstractFloat)
    df = DataFrame() # init empty DataFrame

    # NRST
    res   = parallel_run(ns, rng, TE=TE, keep_xs=false, verbose=false);
    tlens = tourlengths(res)
    nvevs = map(tr -> get_nvevals(tr,ns.np.nexpls), res.trvec)
    TE    = res.toureff[end]
    saveres!(df, "NRST", tlens, nvevs, TE)

    # inputs used for ideal processes
    N = ns.np.N
    R = NRST.rejrates(res)
    Λ = sum(NRST.averej(R))
    ntours = NRST.get_ntours(res)

    # BouncyMC: perfect tuning
    tlens, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so its not meaningful
    saveres!(df, "DTPerf", tlens, nvevs, TE)

    # BouncyMC: actual rejections
    tlens, vNs = run_tours!(BouncyMC(R), ntours)
    TE    = TE_est(vNs)
    nvevs = -1 # technically infty so its not meaningful
    saveres!(df, "DTAct", tlens, nvevs, TE)

    # add other metadata
    insertcols!(df, :N => N)
    insertcols!(df, :Lambda => Λ)

    return df
end

# compute number of V evaluations per tour. assume 1 per explorer step
function get_nvevals(tr::NRST.NRSTTrace{T,TI}, nexpls::AbstractVector) where {T,TI<:Int}
    sum(ip -> ip[1]>zero(TI) ? nexpls[ip[1]] : one(TI), tr.trIP)
end

# store results into df
function saveres!(df::AbstractDataFrame, proc, tlens, nvevs, TE)
    append!(df,
        DataFrame(
            proc=proc, rtser=sum(tlens), rtpar=maximum(tlens),
            costser=sum(nvevs), costpar=maximum(nvevs), TE=TE
        )
    )
end
