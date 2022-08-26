###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

function ess_versus_cost(
    ns::NRSTSampler{T,TI,TF},
    rng::AbstractRNG,
    ntours::Int = 2^14,
    nreps::Int  = 30
    ) where {T,TI,TF}
    N  = ns.np.N
    df = DataFrame() # init empty DataFrame

    # run all the models multiple times, collecting necessary summaries
    for rep in 1:nreps
        print("Iteration $rep/$nreps...")
        # NRST
        res   = parallel_run(ns, rng, ntours=ntours, keep_xs=false, verbose=false);
        tlens = tourlengths(res)
        nvevs = map(tr -> NRST.get_nvevals(tr,ns.np.nexpls), res.trvec)
        TE    = res.toureff[end]
        saveres!(df, "NRST", rep, tlens, nvevs, TE)

        # inputs used for ideal processes
        R = res.rpacc ./ res.visits
        Λ = sum((R[1:(end-1),1] + R[2:end,2]))/2

        # BouncyMC: perfect tuning
        tlens, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
        TE    = (sum(vNs) ^ 2) / (ntours*sum(abs2, vNs))
        nvevs = -1 # technically infty so its not meaningful
        saveres!(df, "DTPerf", rep, tlens, nvevs, TE)

        # BouncyMC: actual rejections
        tlens, vNs = run_tours!(BouncyMC(R), ntours)
        TE = (sum(vNs) ^ 2) / (ntours*sum(abs2, vNs))
        nvevs = -1 # technically infty so its not meaningful
        saveres!(df, "DTAct", rep, tlens, nvevs, TE)

        println("done!")
    end
    return df
end

function saveres!(df::AbstractDataFrame, proc, rep, tlens, nvevs, TE)
    append!(df,
        DataFrame(
            proc=proc,rep=rep,
            cstlen=cumsum(tlens), cmtlen=accumulate(max, tlens),
            csnvev=cumsum(nvevs), cmnvev=accumulate(max, nvevs),
            cESS=TE*(1:length(tlens))
        )
    )
end
