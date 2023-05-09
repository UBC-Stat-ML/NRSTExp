using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

using CubicSplines, Plots, ColorSchemes

# define and tune an NRSTSampler as template
tm        = Banana();#XYModel(3);#ChalLogistic();#MvNormalTM(3,2.,2.);
rng       = SplittableRandom(9406)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    γ = 2.,
    maxcor = 0.95,
    adapt_nexpls = true,
);
cfun    = CubicSpline(ns.np.betas, ns.np.c);

N       = 512
ns_temp = NRSTSampler(tm,rng,N = N,tune=false,nexpl=1)[1]; # template sampler
sh      = SH16Sampler(ns_temp);
fbdr    = FBDRSampler(ns_temp);
plot(b -> cfun(b), 0, 1, label="NRST", palette=:tol_light)
betas = [0.;10. .^ range(-16,0,N)]
NRST.tune!(sh, rng, betas=betas, min_visits=32);
plot!(sh.np.betas, sh.np.c .- sh.np.c[begin], label="SH16")
NRST.tune!(fbdr, rng, betas=betas);
plot!(fbdr.np.betas, fbdr.np.c .- fbdr.np.c[begin], label="FBDR")
title!(string(typeof(tm).name.wrapper) * " (N = $(sh.np.N))")

###############################################################################

using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm = ThresholdWeibull();
rng = SplittableRandom(5427)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    # use_mean = false,
    γ = 2.,
    # nexpl  = 10,
    maxcor = 0.95,
    adapt_nexpls = true,
    # max_rounds=18
);
res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);



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
#     exp=benchmark  \
#     mod=MRNATrans  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.5  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=40322

###############################################################################
# end
###############################################################################


