###############################################################################
# compare TourEff/runtime between NRST and IdealProcesses
###############################################################################

#######################################
# utilities
#######################################

function store_vals!(tup, times, cuTE)
    xs, xp, y = tup
    st = mt = zero(eltype(times))
    for (i, t) in enumerate(times)
        st += t
        if t > mt                    # only store data for the jump events in the max-process
            mt  = t
            push!(xs, log10(st))
            push!(xp, log10(mt))
            push!(y, log10(cuTE[i]))
        end
    end
end

#######################################
# experiment
#######################################

function ess_versus_cost(
    ns::NRSTSampler{T,TI,TF},
    rng::AbstractRNG,
    ntours::Int = 5000,
    nreps::Int  = 30
    ) where {T,TI,TF}
    N      = ns.np.N
    labels = ["NRST", "CT", "DTPerf", "DTAct"]
    dres   = Dict(
        l => (
            xs=TF[], # serial log-times
            xp=TF[], # parallel log-times
            y =TF[]  # cumulative toureffs
            ) for l in labels
    );

    # run all the models multiple times, collecting necessary summaries
    for rep in 1:nreps
        print("Iteration $rep/$nreps...")
        # NRST
        res    = parallel_run(ns, rng, ntours=ntours, keep_xs=false, verbose=false);
        tourls = tourlengths(res)
        cuTE   = res.toureff[end]*(1:ntours)
        store_vals!(dres["NRST"], tourls, cuTE)

        # inputs used for ideal processes
        R = res.rpacc ./ res.visits
        Λ = sum((R[1:(end-1),1] + R[2:end,2]))/2

        # BouncyPDMP
        tourls, vNs  = run_tours!(BouncyPDMP(Λ), ntours)
        TE           = (sum(vNs) ^ 2) / (ntours*sum(abs2, vNs))
        cuTE        = TE*(1:ntours)
        scale_tourls = N*tourls .+ 2.           # the shortest tour has n(0)=2 and the perfect roundtrip has n(2)=2N+2
        store_vals!(dres["CT"], scale_tourls, cuTE)

        # BouncyMC: perfect tuning
        tourls, vNs = run_tours!(BouncyMC(Λ/N,N), ntours)
        TE = (sum(vNs) ^ 2) / (ntours*sum(abs2, vNs))
        cuTE = TE*(1:ntours)
        store_vals!(dres["DTPerf"], tourls, cuTE)

        # BouncyMC: actual rejections
        tourls, vNs = run_tours!(BouncyMC(R), ntours)
        TE = (sum(vNs) ^ 2) / (ntours*sum(abs2, vNs))
        cuTE = TE*(1:ntours)
        store_vals!(dres["DTAct"], tourls, cuTE)
        println("done!")
    end

    # convert to DataFrame and return
    df = mapreduce(((k,v),) -> insertcols!(DataFrame(v), :proc => k), append!, dres)
    return df
end

