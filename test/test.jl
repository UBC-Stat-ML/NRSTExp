using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using NRSTExp.CompetingSamplers
using SplittableRandoms

# define and tune an NRSTSampler as template
const tm = MvNormalTM(32,4.,2.);
rng = SplittableRandom(40322)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);
res=parallel_run(ns,rng,NRST.NRSTTrace(ns),TE=TE);

TE_ele = inv(1+2Λone)
#TE_ele = inv(1+2Λ)
nrpt = NRST.NRPTSampler(ns);
trVs = NRST.collectVsSerial!(nrpt, rng, 2^16);
maxcor = 0.001
NRST.tune_nexpls!(ns.np.nexpls, trVs, maxcor)
plot(ns.np.nexpls)
ns.np.nexpls
res=parallel_run(ns,rng,ntours=2048);
last(res.toureff)


using NRSTExp.IdealIndexProcesses
NRSTExp.get_TE(last(run_tours!(BouncyPDMP(Λone/2), 2^22)))



using Random
using SplittableRandoms: SplittableRandom, split
import Base.Threads.@threads

println("Number of threads: $(Threads.nthreads())")

const n_iters = 10000;
const master_rng = SplittableRandom(1)
result = zeros(n_iters);
rngs = [split(master_rng) for _ in 1:n_iters]
@threads for i in 1:n_iters
    # in a real problem, do some expensive calculation here...
    result[i] = rand(rngs[i]);
end
println("Result: $(last(result))")

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=mean    \
#     cor=0.95 \
#     gam=2.5  \
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


