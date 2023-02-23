using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(1111111)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    NRST.SliceSamplerSteppingOut,
    use_mean   = true,
    maxcor     = 0.9,
    γ          = 2.0,
    xpl_smooth_λ = 1e-5,
);
res=parallel_run(ns,rng,NRST.NRSTTrace(ns),TE=TE,δ=0.5);
sin_res = NRST.inference_on_V(res,h=sin)

using Plots
using Plots.PlotMeasures: px
plots = NRST.diagnostics(ns, res);
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.9 \
#     gam=2.0  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=1111

###############################################################################
# end
###############################################################################


