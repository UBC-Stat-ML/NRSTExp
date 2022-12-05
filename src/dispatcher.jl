###############################################################################
# parses the ARGS from command line and executes an experiment
###############################################################################

# for calling from the command line
function dispatch()
    isinteractive() && error("This method can only be called in non-interactive mode.")
    println("This is NRSTExp! Parsing parameters...")
    pars = Dict(map(s -> eachsplit(s, "="), ARGS))
    display(pars)
    println("\nLaunching experiment...")
    dfres = dispatch(pars)
    
    # write data
    println("\nNRSTExp: experiment finished successfully!")
    print("\tWriting metadata...")
    fn = "NRSTExp_" * string(hash(join(ARGS)), base = 16)
    open(fn * ".tsv", "w") do io
        writedlm(io, pars)
    end
    print("done!\n\tWriting data...")
    CSV.write(fn * ".csv.gz", dfres, compress=true) # use gzip compression
    println("done!\nGood bye!")
    return
end

function dispatch(pars::Dict)
    exper   = pars["exp"]
    model   = pars["mod"]
    maxcor  = parse(Float64, pars["cor"])
    γ       = parse(Float64, pars["gam"])
    usemean = (pars["fun"] == "mean")
    xplsmth = parse(Bool, pars["xps"])
    rseed   = parse(Int, pars["seed"])
    rng     = SplittableRandom(rseed)

    # load model. should at least produce a TemperedModel
    need_build = true
    if model == "MvNormal"
        tm = MvNormalTM(32,4.,2.)
        if usemean
            # do special tuning with exact free_energy
            ns, TE, Λ = NRSTSampler(
                tm,
                rng,
                use_mean   = usemean,
                maxcor     = maxcor,
                γ          = γ,
                xpl_smooth = xplsmth,
                do_stage_2 = false
            )
	        copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy
	        need_build = false
        end
    elseif model == "XYModel_small"
        tm = XYModel(5)
    elseif model == "XYModel_big"
        tm = XYModel(8)
    elseif model == "HierarchicalModel"
        tm = HierarchicalModel()
    elseif model == "Challenger"
        tm = ChalLogistic()
    elseif model == "MRNATrans"
        tm = MRNATrans()
    elseif model == "Titanic"
        tm = TitanicHS()
    else
        throw(ArgumentError("Model $model not yet implemented."))
    end

    # build and tune sampler
    if need_build
        ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            use_mean   = usemean,
            maxcor     = maxcor,
            γ          = γ,
            xpl_smooth = xplsmth
        )
    end

    # dispatch experiment
    # should return a dataframe that we can then save as csv
    if exper == "hyperparams"
        dfres = hyperparams(ns, rng, TE)
    elseif exper == "benchmark"
        dfres = benchmark(ns, rng, TE)
    else
        throw(ArgumentError("Experiment $exper not yet implemented."))
    end
    return dfres
end
