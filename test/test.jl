using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm = HierarchicalModel();#ThresholdWeibull();#Funnel();#Banana();#XYModel(8);#MRNATrans();#ChalLogistic();#MvNormalTM(32,4.,2.);
rng = SplittableRandom(40378)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res = parallel_run(ns, rng, TE=TE);
println("NRST cost: $(sum(NRST.get_nvevals, res.trvec))")
res = parallel_run(ns, rng, NRST.NRSTTrace(ns),TE=TE);
vmin,imin = findmin(res.trVs[end])
xmin = res.xarray[end][imin]

using Plots

f0 = 0.#free_energy(tm,0);
# plot(b->free_energy(tm,b),0,1,label="true")
scatter(ns.np.betas,ns.np.c .+ (f0-first(ns.np.c)),label="NRST")
sh = NRST.init_sampler(SH16Sampler, tm, rng, N=512-1, xv_init=(xmin,vmin)); # bottleneck is HierarchicalModel
mvs = NRST.tune!(sh,rng,max_steps=2^18);
scatter!(sh.np.betas,sh.np.c .+ (f0-first(sh.np.c)),label="SH16")
TE_sh = last(parallel_run(sh, rng, ntours=2048).toureff)
res_sh = parallel_run(sh, rng, TE=TE_sh);
println("SH16 cost: $(sum(NRST.get_nvevals, res_sh.trvec))")


fbdr = NRST.init_sampler(FBDRSampler, tm, rng, N=512-1); # bottleneck for N is Funnel
NRST.tune!(fbdr,rng,nsteps=2^7, log_grid=true);
scatter!(fbdr.np.betas,fbdr.np.c .+ (f0-first(fbdr.np.c)),label="FBDR")
TE_fbdr = last(parallel_run(fbdr, rng, ntours=2048).toureff)
res_fbdr = parallel_run(fbdr, rng, TE=TE_fbdr);
println("FBDR cost: $(sum(NRST.get_nvevals, res_fbdr.trvec))")

using OnlineStats
value.(mvs)


# TODO: convert this into a viz method in NRSTExp
# using Plots, ColorSchemes
# DEF_PAL = ColorSchemes.seaborn_colorblind

# # utility for creating the Λ plot
# function plot_lambda(Λ,bs,lab)
#     c1 = DEF_PAL[1]
#     c2 = DEF_PAL[2]
#     p = plot(
#         Λ, 0., 1., label = "", legend = :bottomright,
#         xlim=(0.,1.), color = c1, grid = false, ylim=(0., Λ(bs[end])),
#         xlabel = "β", ylabel = "Λ(β)"
#     )
#     plot!(p, [0.,0.], [0.,0.], label=lab, color = c2)
#     for (i,b) in enumerate(bs[2:end])
#         y = Λ(b)
#         plot!(p, [b,b], [0.,y], label="", color = c2)                  # vertical segments
#         plot!(p, [0,b], [y,y], label="", color = c1, linestyle = :dot) # horizontal segments
#     end
#     p
# end

# # Lambda Plot
# betas = ns.np.betas
# averej = NRST.averej(res)
# f_Λnorm, _, Λs = NRST.gen_lambda_fun(betas, averej, ns.np.log_grid)
# plot_lambda(β->((x = ns.np.log_grid ? NRST.floorlog(β) : β);Λs[end]*f_Λnorm(x)),betas,"")

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=TitanicHS  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchOwnTune  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.0  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=40322

###############################################################################
# end
###############################################################################


