using Pkg; Pkg.update()
using NRST
using NRST.ExamplesGallery
using NRSTExp

tm  = MvNormalTM(32,4.,2.)
rng = SplittableRandom(20191018)
ns, ts = NRSTSampler(
    tm,
    rng,
    N = 12,
    verbose = true,
    do_stage_2 = false,
    maxcor = 0.8
);
copyto!(ns.np.c, free_energy(tm, ns.np.betas)); # use optimal tuning
dfres = ess_versus_cost(ns, rng)