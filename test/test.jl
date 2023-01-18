using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = TitanicHS()
rng = SplittableRandom(44697)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    xpl_smooth_λ = 0.1
);

# 0.002775895032825842
using Plots

lσs = [log(first(pars)) for pars in ns.np.xplpars]
plot(lσs)
plot(ns.np.nexpls)
using Distributions
nst = NRSTExp.ExamplesGallery.NSTDist(3.,10.,3.0)
lst = 10. + TDist(3.)*3.
tlst = truncated(lst, lower=0.)
typeof(tlst)
logpdf(nst, -1.0)
logpdf(tlst, -1.0)

logpdf(NoncentralT(3.,10.), 1.0)
###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.6 \
#     gam=25.0  \
#     xps=0.1 \
#     seed=1111

###############################################################################
# end
###############################################################################
