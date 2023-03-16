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
    println("done!\nGoodbye!")
    return
end

function dispatch(pars::Dict)
    nexpl   = -1
    exper   = pars["exp"]
    model   = pars["mod"]
    maxcor  = parse(Float64, pars["cor"])
    if maxcor >= 1.0                      # hack to pass nexpl instead of maxcor
        nexpl = round(Int, maxcor)
    end
    γ       = parse(Float64, pars["gam"])
    usemean = (pars["fun"] == "mean")
    TXpl    = pars["xpl"] == "SSSO" ? NRST.SliceSamplerSteppingOut : NRST.SliceSamplerDoubling
    xplsmλ  = parse(Float64, pars["xps"])
    rseed   = parse(Int, pars["seed"])
    rng     = SplittableRandom(rseed)

    # load model. should at least produce a TemperedModel
    if model == "MvNormal"
        tm = MvNormalTM(32,4.,2.)
    elseif model == "XYModel"
        tm = XYModel(8)
    elseif model == "HierarchicalModel"
        tm = HierarchicalModel()
    elseif model == "Challenger"
        tm = ChalLogistic()
    elseif model == "MRNATrans"
        tm = MRNATrans()
    elseif model == "Titanic"
        tm = Titanic()
    else
        throw(ArgumentError("Model $model not yet implemented."))
    end

    # build and tune sampler
    ns, TE, Λ = NRSTSampler(
        tm,
        rng,
        TXpl,
        use_mean   = usemean,
        maxcor     = maxcor,
        nexpl      = nexpl,
        γ          = γ,
        xpl_smooth_λ = xplsmλ,
        adapt_nexpls = nexpl < 0
    )

    # dispatch experiment
    # should return a dataframe that we can then save as csv
    if exper == "hyperparams"
        dfres = hyperparams(ns, rng, TE, Λ)
    elseif exper == "benchmark"
        dfres = benchmark(ns, rng, TE, Λ)
    else
        throw(ArgumentError("Experiment $exper not yet implemented."))
    end
    return dfres
end
