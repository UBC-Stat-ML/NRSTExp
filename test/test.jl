using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(44697)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
);

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

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=benchmark  \
#     mod=XYModel  \
#     fun=mean    \
#     cor=0.6 \
#     gam=25.0  \
#     xps=0.1 \
#     seed=24482

###############################################################################
# end
###############################################################################
