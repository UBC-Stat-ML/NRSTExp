###############################################################################
# parses the ARGS from command line and executes an experiment
# it uses positional arguments
#   1) experiment
#   2) model
#   3) maxcor
# TODO: need a "setup" script before running this script that 
#   - installs latest NRST version: just use "add", it is smart (does nothing if no changes)
#   - instantiates the environment
###############################################################################

opt_N(Λ) = ceil(Int, Λ*(1+sqrt(1 + inv(1+2Λ))))

function dispatch()
    # parse arguments
    exper  = ARGS[1]
    model  = ARGS[2]
    maxcor = parse(Float64, ARGS[3])

    # build model
    # should at least produce a TemperedModel
    rng = SplittableRandom(0x0123456789abcdfe)
    need_build = true
    if model == "mvNormals"
        tm = MvNormalTM(32,4.,2.)
        Λ  = 5.3 # best estimate of true barrier        
        # do special tuning with exact free_energy
        N  = opt_N(Λ) 
        ns, ts = NRSTSampler(
            tm,
            rng,
            N = N,
            verbose = true,
            do_stage_2 = false,
            maxcor = maxcor
        )
        copyto!(ns.np.c, free_energy(tm, ns.np.betas)); # use optimal tuning
        need_build = false
    else
        throw(ArgumentError("$model not yet implemented."))
    end

    # build and tune sampler
    if need_build
        N  = opt_N(Λ)
        ns, ts = NRSTSampler(
            tm,
            rng,
            N = N,
            verbose = true,
            maxcor = maxcor
        )
    end

    # dispatch experiment
    if exper == "ess_versus_cost"
        dfres = ess_versus_cost(ns, rng)
    else
        throw(ArgumentError("$model not yet implemented."))
    end

    # write data
    fn = "$(exper)_$(model)_$(round(maxcor,digits=2)).csv"
    mkdir("output")
    fp = joinpath("output", fn)
    CSV.write(fp, dfres)

    return
end