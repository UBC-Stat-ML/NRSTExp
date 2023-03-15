using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using NRSTExp.CompetingSamplers
using SplittableRandoms

# define and tune an NRSTSampler as template
const tm  = MRNATrans()
rng = SplittableRandom(40322)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean=true,
    γ=2.5,
    maxcor=2
);

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
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


