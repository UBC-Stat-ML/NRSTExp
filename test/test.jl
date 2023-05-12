using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
#HierarchicalModel();#ThresholdWeibull();
tm = Funnel();#XYModel(8);#MRNATrans();#Banana();#ChalLogistic();#MvNormalTM(32,4.,2.);
rng = SplittableRandom(5427)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res = parallel_run(ns, rng, TE=TE);
println("NRST cost: $(sum(NRST.get_nvevals, res.trvec))")

using Plots

f0 = 0.#free_energy(tm,0);
# plot(b->free_energy(tm,b),0,1,label="true")
scatter(ns.np.betas,ns.np.c .+ (f0-first(ns.np.c)),label="NRST")
# sh = NRST.init_sampler(SH16Sampler, tm, rng, N=512-1);
# mvs = NRST.tune!(sh,rng);
# scatter!(sh.np.betas,sh.np.c .+ (f0-first(sh.np.c)),label="SH16")
fbdr = NRST.init_sampler(FBDRSampler, tm, rng, N=512-1);
NRST.tune!(fbdr,rng,nsteps=2^7);
scatter!(fbdr.np.betas,fbdr.np.c .+ (f0-first(fbdr.np.c)),label="FBDR")

TE_sh = last(parallel_run(sh, rng, ntours=512).toureff)
res_sh = parallel_run(sh, rng, TE=TE_sh);
println("SH16 cost: $(sum(NRST.get_nvevals, res_sh.trvec))")

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
#     exp=benchSH16tune  \
#     mod=Banana  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.0  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=40322

###############################################################################
# end
###############################################################################


