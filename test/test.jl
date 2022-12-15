using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

tm  = ChalLogistic();##MRNATrans()#XYModel(6)#HierarchicalModel()##XYModel(12)##MRNATrans()##XYModel(8)##
rng = SplittableRandom(40322)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    γ=16.0,
    max_rounds=8
);


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
