using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm = Banana();
rng = SplittableRandom(2911)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    # use_mean = false,
    γ = 4.,
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
#     exp=hyperparams  \
#     mod=ThresholdWeibull  \
#     fun=median    \
#     cor=0.95 \
#     gam=11  \
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


