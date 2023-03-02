using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(5040)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean=true,
    γ=2.5,
    maxcor=0.95
);
res=parallel_run(ns,rng,TE=TE);
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
#     exp=benchmark  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.5  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=1111

###############################################################################
# end
###############################################################################


