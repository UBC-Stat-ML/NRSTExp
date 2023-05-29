using NRSTExp
NRSTExp.get_barrier_df()

using NRST
using NRSTExp
using NRSTExp.CompetingSamplers
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm = ChalLogistic();#Funnel();#ThresholdWeibull();#HierarchicalModel();#Banana();#XYModel(8);#MRNATrans();#MvNormalTM(32,4.,2.);
rng = SplittableRandom(89440)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);

using Plots
tms = (Banana(), Funnel(), HierarchicalModel(), MRNATrans(), ThresholdWeibull(), XYModel(8));
labs = ("Banana", "Funnel", "HierarchModel", "MRNATrans", "ThreshWeibull", "XYModel")
df = DataFrame()





vcat
# rho' Plot
ρ_prime = NRSTExp.build_rhoprime(f_Λnorm, Λs[end], ns.np.log_grid);
plot!(ρ_prime,0,1, ylims=(0,Λs[end]))



spl = fit(SmoothingSpline, X, Y, 250.0) # λ=250.0
Ypred = predict(spl) # fitted vector

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
#     sam=GT95 \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.0  \
#     xpl=SSSO \
#     xps=1e-5 \
#     seed=89440

###############################################################################
# end
###############################################################################


