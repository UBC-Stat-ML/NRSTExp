using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using Plots
using Plots.PlotMeasures: px

tm  = HierarchicalModel()
rng = SplittableRandom(4)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res   = parallel_run(ns, rng, TE=TE, keep_xs=false);
plots = NRST.diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)
