###############################################################################
# parses the ARGS from command line and executes an experiment
# it uses positional arguments
#   1) experiment
#   2) model
#   3) maxcor
# TODO: need a "setup" script before running this script that 
#   - pulls latest changes for our repos: easiest way is to simply Pkg.update()
#   - this strategy requires pushing Manifest.toml, which in general is not 
#     advisable but it simplifies greatly our situation
# TODO: put all tests inside a Nextflow script. use Alex's tutorial in
#   https://github.com/UBC-Stat-ML/nextflow-notes
###############################################################################

opt_N(Λ) = ceil(Int, Λ*(1+sqrt(1 + inv(1+2Λ))))

# for calling from the command line
function dispatch()
    # parse arguments
    exper  = ARGS[1]
    model  = ARGS[2]
    maxcor = parse(Float64, ARGS[3])
    dfres  = dispatch(exper,model,maxcor)
    
    # write data
    od = mkdir("output")
    fn = "E:$(exper)_M:$(model)_MC:$(round(maxcor,digits=2)).csv"
    fp = joinpath(od, fn)
    CSV.write(fp, dfres)
    return
end

function dispatch(
    exper::String,
    model::String,
    maxcor::AbstractFloat
    )
    rng = SplittableRandom(0x0123456789abcdfe) # seed the (p)rng

    # load model
    # should at least produce a TemperedModel
    need_build = true
    if model == "MvNormal"
        tm = MvNormalTM(32,4.,2.)
        Λ  = 5.32 # best estimate of true barrier        

        # do special tuning with exact free_energy
        N = opt_N(Λ)
        ns, ts = NRSTSampler(
            tm,
            rng,
            N = N,
            verbose = true,
            do_stage_2 = false,
            maxcor = maxcor
        )
        copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy
        need_build = false
    elseif model == "XYModel"
        tm = XYModel(8)
        Λ  = 5.25 # best estimate of true barrier        
    elseif model == "HierarchicalModel"
        tm = HierarchicalModel()
        Λ  = 4.7 # best estimate of true barrier        
    else
        throw(ArgumentError("Model $model not yet implemented."))
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
    # should return a dataframe that we can then save as csv
    if exper == "ess_versus_cost"
        dfres = ess_versus_cost(ns, rng)
    else
        throw(ArgumentError("Experiment $exper not yet implemented."))
    end
    return dfres
end
