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
    fn = "NRSTExp_$(pars["seed"])_" * Dates.format(Dates.now(), "yyyymmddHHMMSSs")
    open(fn * ".tsv", "w") do io
        writedlm(io, pars)
    end
    CSV.write(fn * ".csv.gz", dfres, compress=true) # use gzip compression
    return
end

function dispatch(pars::Dict)
    exper   = pars["exp"]
    model   = pars["mod"]
    maxcor  = parse(Float64, pars["cor"])
    γ       = parse(Float64, pars["gam"])
    usemean = (pars["fun"] == "mean")
    rseed   = parse(Int, pars["seed"])
    rng     = SplittableRandom(rseed)

    # load model. should at least produce a TemperedModel
    need_build = true
    if model == "MvNormal"
        # do special tuning with exact free_energy
        tm = MvNormalTM(32,4.,2.)
        ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            use_mean   = usemean,
            maxcor     = maxcor,
            γ          = γ,
            do_stage_2 = false
        )
        copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy
        need_build = false
    elseif model == "XYModel"
        tm = XYModel(8)
    elseif model == "HierarchicalModel"
        tm = HierarchicalModel()
    elseif model == "Challenger"
        tm = ChalLogistic()
    elseif model == "Transfection"
        tm = MRNATransTuring()
    else
        throw(ArgumentError("Model $model not yet implemented."))
    end

    # build and tune sampler
    if need_build
        ns, TE, Λ = NRSTSampler(
            tm,
            rng,
            use_mean = usemean,
            maxcor   = maxcor,
            γ        = γ
        )
    end

    # dispatch experiment
    # should return a dataframe that we can then save as csv
    if exper == "benchmark"
        dfres = benchmark(ns, rng, TE)
    else
        throw(ArgumentError("Experiment $exper not yet implemented."))
    end
    return dfres
end
