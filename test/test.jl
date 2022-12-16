using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = TitanicHS()
rng = SplittableRandom(753)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    γ=14.,
    maxcor=0.6,
)
using Plots

lσs = [log(first(pars)) for pars in ns.np.xplpars]
plot(lσs)
findmax(ns.np.nexpls)
###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.9 \
#     gam=8.0  \
#     xps=1e-5 \
#     seed=1111

###############################################################################
# end
###############################################################################
