using NRSTExp

push!(ARGS, "ess_versus_cost")
push!(ARGS, "HierarchicalModel")
push!(ARGS, "0.99")
dispatch()
# julia -t 4 --project -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99
# julia --project -e "using NRSTExp" ess_versus_cost HierarchicalModel 0.99
# ./julia -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99

using NRSTExp.ExamplesGallery
using NRST

tm = ChalLogistic();
rng = SplittableRandom(1312)
ns, ts = NRSTSampler(
    tm,
    rng,
    N = 10,
    verbose = true
);
res   = parallel_run(ns, rng, ntours = ts.ntours)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)
