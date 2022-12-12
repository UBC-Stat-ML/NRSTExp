using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using Random
using SplittableRandoms

tm  = TitanicHS()#MRNATrans()#XYModel(8)#HierarchicalModel()#
rng = SplittableRandom(227)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    xpl_smooth_λ=3,
);

using Plots

lσs  = [log(first(pars)) for pars in ns.np.xplpars];
plot(lσs)

ens = NRST.replicate(ns.xpl, ns.np.betas);
for (i,xpl) in enumerate(ens)
    xpl.sigma[] = randexp()
end
NRST.tune_explorers!(ns.np, ens, rng,smooth_λ=0);
lσs  = [log(first(NRST.params(xpl))) for xpl in ens];
plot(lσs)
plσs = running_median(lσs, 11,:asymmetric_truncated) 
for (i,xpl) in enumerate(ens)
    xpl.sigma[] = exp(plσs[i])
end
plot!(plσs)

using FastRunningMedian

plot!(running_median(lσs, 7))

extrema(σs)
plot(ns.np.nexpls)
fbdr = FBDRSampler(ns);
NRST.tour!(fbdr,rng)
ntours = 2048# NRST.min_ntours_TE(TE);
res = parallel_run(fbdr,rng,ntours=ntours);

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
