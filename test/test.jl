using NRSTExp

push!(ARGS, "ess_versus_cost")
push!(ARGS, "HierarchicalModel")
push!(ARGS, "0.99")
dispatch()
# JULIA_PKG_USE_CLI_GIT=true julia --project -e "using Pkg; Pkg.update()"
# julia -t 4 --project -e "using NRSTExp; dispatch()" ess_versus_cost Challenger 0.99
# julia --project -e "using NRSTExp" ess_versus_cost HierarchicalModel 0.99
# ./julia -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99

using Plots
using Plots.PlotMeasures: px
using ColorSchemes: seaborn_colorblind
using StatsBase
using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

rng = SplittableRandom(0x0123456789abcdfe)
tm = XYModel(8)
Λ  = 5.25
N  = NRSTExp.opt_N(Λ)
maxcor = 0.8
ns = NRSTSampler(
            tm,
            rng,
            N = N,
            verbose = true,
            maxcor = maxcor
        )
res   = parallel_run(ns, rng, ntours = 32768)
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/XY_model/diags.png)

# ## Notes on the results
# ### Bivariate density plots of two neighbors
#
# Note that for ``\theta_a,\theta_b \in [-\pi,\pi]``,
# ```math
# \cos(\theta_a-\theta_b) = 0 \iff \theta_a - \theta_b = 2k\pi
# ```
# for ``k\in\{-1,0,1\}``. The plots below show that as ``\beta`` increases,
# the samples concentrate at either of three loci, each described by a different
# value of ``k``. Indeed, the diagonal corresponds to ``k=0``, while the
# off-diagonal loci have ``|k|=1``. In the ideal physical model,
# ``\theta_s \in (-\pi,\pi]``, so the non-coherent states have 0 prior probability.
# In floating-point arithmetic, however, the distinction between open and closed
# is impossible.
ngrid     = 50
nsub_wish = ngrid*ngrid*5
parr      = []
for (i,xs) in enumerate(res.xarray)
    # i=1; xs=res.xarray[i]
    β      = ns.np.betas[i]
    nsam   = length(xs)
    nsub   = min(nsub_wish,nsam)
    idx    = sample(1:nsam, nsub, replace=false, ordered=true)
    X      = hcat([x[1:2] for x in xs[idx]]...)
    plev   = scatter(
        X[1,:], X[2,:], markeralpha=min(1., max(0.08, 1000/nsub)), 
        palette=seaborn_colorblind,
        title="β=$(round(β,digits=2))", label=""
    )
    push!(parr, plev)
end
N  = ns.np.N
nc = min(N+1, ceil(Int,sqrt(N+1)))
nr = ceil(Int, (N+1)/nc)
for i in (N+2):(nc*nr)
    push!(parr, plot())
end 
pcover = plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr), ticks=false, 
    showaxis = false, legend = false, colorbar = false
)


###############################################################################
###############################################################################

using NRST, NRSTExp, NRSTExp.ExamplesGallery
using Plots
using Plots.PlotMeasures: px
using ColorSchemes: okabe_ito

opt_N = NRSTExp.opt_N
maxcor = 0.8
rng = SplittableRandom(1313)
tm = MvNormalTM(32,4.,2.)
Λ  = 5.32 # best estimate of true barrier
N = opt_N(Λ)
ns = NRSTSampler(tm, rng, N = N, verbose = true, maxcor = maxcor);
copyto!(ns.np.c, free_energy(tm, ns.np.betas)) # use exact free energy
res   = parallel_run(ns, rng, ntours=2^17, keep_xs=false);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/mvNormals/diags.png)

# ## Distribution of the potential
# We compare the sample distribution of (properly scaled) ``V(x)`` obtained 
# using various sampling strategies against the analytic distribution.
ntours = 32768
sbsq(b)= NRSTExp.ExamplesGallery.sbsq(tm, b)
xpls   = NRST.replicate(ns.xpl, ns.np.betas);
trVs   = NRST.collectVs(ns.np, xpls, rng, ceil(Int, sum(ns.np.nexpls)/ns.np.N)*ntours);
resser = NRST.SerialRunResults(NRST.run!(ns, rng, nsteps=2*ns.np.N*ntours));
restur = NRST.run_tours!(ns, rng, ntours=ntours, keep_xs=false);
resPT  = NRST.rows2vov(NRST.run!(NRST.NRPTSampler(ns),rng,ntours).Vs);
parr   = []
for (i,trV) in enumerate(trVs)
    β     = ns.np.betas[i]
    κ     = (2/sbsq(β))    # scaling factor
    sctrV = κ .* trV
    p = plot(
        get_scaled_V_dist(tm,β), label="True", palette=okabe_ito,
        title="β=$(round(β,digits=2))"
    )
    density!(p, sctrV, label="IndExps", linestyle =:dash)
    sctrV = κ .* resPT[i]
    density!(p, sctrV, label="NRPT", linestyle =:dash)
    sctrV = κ .* resser.trVs[i]
    density!(p, sctrV, label="SerialNRST", linestyle =:dash)
    sctrV = κ .* restur.trVs[i]
    density!(p, sctrV, label="TourNRST", linestyle =:dash)
    sctrV = κ .* res.trVs[i]
    density!(p, sctrV, label="pTourNRST", linestyle =:dash)
    push!(parr, p)
end
N  = ns.np.N
nc = min(N+1, ceil(Int,sqrt(N+1)))
for i in (N+2):(nc*nr)
    push!(parr, plot(ticks=false, showaxis = false, legend = false))
end
pdists = plot(
    parr..., layout = (nr,nc), size = (300*nc,333*nr)
)


truc = free_energy(tm, ns.np.betas)
truc .-= truc[1]
copyto!(ns.np.c, truc) # use exact free energy
NRST.STEPSTONE_FWD_WEIGHT[] = 1.
NRST.tune_c!(ns.np, parallel_run(ns, rng, ntours=2 ^ 17, keep_xs=false, verbose=false))
plot(ns.np.c .- truc, label="w=1.")
copyto!(ns.np.c, truc) # use exact free energy
NRST.STEPSTONE_FWD_WEIGHT[] = 0.
NRST.tune_c!(ns.np, parallel_run(ns, rng, ntours=2 ^ 17, keep_xs=false, verbose=false))
plot!(ns.np.c .- truc, label="w=0.")
copyto!(ns.np.c, truc) # use exact free energy
NRST.STEPSTONE_FWD_WEIGHT[] = 0.5
NRST.tune_c!(ns.np, parallel_run(ns, rng, ntours=2 ^ 17, keep_xs=false, verbose=false))
plot!(ns.np.c .- truc, label="w=0.5")


###############################################################################
###############################################################################

using Plots
using Plots.PlotMeasures: px
using NRST
using NRSTExp
using NRSTExp.ExamplesGallery

Λ     = 4.7
N     = NRSTExp.opt_N(Λ)
rng   = SplittableRandom(123) # seed the (p)rng
tm    = HierarchicalModel()
ns    = NRSTSampler(tm, rng, N = N, verbose = true);
res   = parallel_run(ns, rng, ntours = 2^14);
plots = diagnostics(ns, res)
hl    = ceil(Int, length(plots)/2)
pdiags=plot(
    plots..., layout = (hl,2), size = (900,hl*333),left_margin = 40px,
    right_margin = 40px
)

#md # ![Diagnostics plots](assets/hierarchical_model/diags.png)

# ## Notes on the results
# ### Inspecting within and between-group std. devs.
X = hcat([exp.(0.5*x[1:2]) for x in res.xarray[end]]...)
pcover = scatter(
    X[1,:],X[2,:], xlabel="τ: between-groups std. dev.",
    markeralpha = min(1., max(0.08, 1000/size(X,2))),
    ylabel="σ: within-group std. dev.", label=""
)
