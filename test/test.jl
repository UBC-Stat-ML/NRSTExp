using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(3000)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean   = true,
    maxcor     = 0.9,
    γ          = 8.0,
    xpl_smooth_λ = 0.1
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
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=HierarchicalModel  \
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=2798

###############################################################################
# end
###############################################################################

###############################################################################
# find best w for SliceSampler
# conclusion: all w achieve same autocor and sqdist, but w=15std(x) is cheapest
###############################################################################

using Distributions, DynamicPPL, Plots, StatsBase
using SplittableRandoms
using NRST

const σ = Ref(1.0)
@model function ToyModel()
    y1 ~ Normal(0., σ[])
end
const tm  = NRST.TuringTemperedModel(ToyModel())
const rng = SplittableRandom(1)
const x   = rand(tm,rng)
const ps  = NRST.potentials(tm,x)
const ws  = 2 .^ range(0,5,30)
const sqds= similar(ws)
const ss  = NRST.SliceSampler(
    tm, x, Ref(1.0), Ref(ps[1]), Ref(0.0), Ref(ps[1])
);

function get_ac_nvs()
    nsim= 10000
    vs  = similar(ws, nsim)
    nvs = 0
    for i in 1:nsim
        nvs  += last(NRST.step!(ss,rng))
        vs[i] = ss.curVref[]
    end
    first(autocor(vs, 1:1)), nvs
end

acs = similar(ws)
nvs = Vector{Int}(undef, length(ws))
for (i,w) in enumerate(ws)
    ss.w[] = w
    print("Set w=$(ss.w[]). Sampling...")
    ac, nv = get_ac_nvs()
    println("done!")
    acs[i]=ac
    nvs[i]=nv
end
_,iopt = findmin(nvs)
plot(ws,nvs)
vline!([ws[iopt]])