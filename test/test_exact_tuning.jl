using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using NRSTExp.CompetingSamplers
using SplittableRandoms
using Plots
using Statistics
using Distributions
using QuadGK
import NRSTExp.ExamplesGallery: free_energy, get_V_dist

const tm = MvNormalTM(32,4.,2.);

# numerical approximation of E^{β}[|V - E^{β}[V]|]
function madV(β, maxV=1000.0)
    distV = get_V_dist(tm, β)
    mV    = mean(distV)
    quadgk(v -> pdf(distV,v)*abs(v-mV), zero(maxV), maxV)
end

# numerical approximation of (1/2)int_0^β db mad(b)
Λfun(β) = first(quadgk(b -> first(madV(b)), zero(β), β))/2
plot(β -> first(madV(β))/2, 0, 1)
plot!(β -> Λfun(β), 0, 1)

# optimal grid based on inversion of Λfun
const Λone = Λfun(1.0)
function get_opt_grid(N)
    opt_grid = zeros(N+1)
    for i in 1:(N-1)
        opt_grid[i+1] = first(NRST.monoroot(β -> (Λfun(β)/Λone - i/N),opt_grid[i],1.0))
    end
    opt_grid[end] = 1.0
    opt_grid
end

# create a sampler without tuning it
TE_ele = inv(1+2Λone)
N      = NRST.optimal_N(Λone, 8)
rng    = SplittableRandom(335566)
ns     = first(NRSTSampler(tm, rng, N = N, tune=false));

# do exact tuning of grid and c
opt_grid = get_opt_grid(N)
opt_c    = free_energy(tm, opt_grid)
copyto!(ns.np.betas,opt_grid)
copyto!(ns.np.c, opt_c)

# collect V samples to use for tuning nexpls
trVs = NRST.collectVsSerial!(NRST.NRPTSampler(ns), rng, 2^16);

# check if ~0 correlation delivers TE under ELE
maxcor = 0.001
NRST.tune_nexpls!(ns.np.nexpls, trVs, maxcor)
extrema(ns.np.nexpls)
res=parallel_run(ns,rng,ntours=NRST.min_ntours_TE(TE_ele,0.95,0.15));
last(res.toureff)


using NRSTExp.IdealIndexProcesses
NRSTExp.get_TE(last(run_tours!(BouncyPDMP(Λone/2), 2^22)))

