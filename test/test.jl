using NRSTExp
NRSTExp.make_runtime_plot()

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


using Random, Plots, ColorSchemes, DataStructures

Λ    = 5.                # tempering barrier
K    = 10             # total number of tours
Prat = 2. .^ range(-4, 0) # vector of proportions of workers over total number of tours


# simulate
rng = Xoshiro(1)
ts  = Λ*randexp(rng, K)
i   = argmax(ts)
if i <= K/2 # swap position of max so that plot looks more natural (see comment above)
    j = K-i+1
    m = ts[i]
    ts[i] = ts[j]
    ts[j] = m
end
cpt = sum(ts)
Pvec= round.(Int, K .* Prat)
pq  = PriorityQueue{Int64,Float64}()
P = 3
rs,ws = NRSTExp.simulate_events(pq,ts,P)
rs

# move time fwd in the queue <=> substract fixed t from all queue priorities
# key: this does not change relative order! So we can modify their setindex!
# method to avoid the percolating phase. See below
# https://github.com/JuliaCollections/DataStructures.jl/blob/d0dd2a012ebd07ff31193b21130109baa50cfe2b/src/priorityqueue.jl#L192
function NRSTExp.move_fwd!(pq::PriorityQueue{K,V},t) where {K,V}
    for key in keys(pq)
        i = pq.index[key]
        oldvalue = pq.xs[i].second
        pq.xs[i] = Pair{K,V}(key, oldvalue-t)
    end
    pq
end


res = [NRSTExp.simulate_events(pq,ts,P) for P in Pvec]

# plot the results
lss = reverse!([:solid, :dash, :dot, :dashdot, :dashdotdot])
pal = ColorSchemes.seaborn_colorblind6
plt = plot(fontfamily = "Helvetica")
for (i,p) in enumerate(Pvec)
    #i=1; p = Prat[i]
    rs,ws = res[i]
    lab   = NRSTExp.fmt_thousands(p)
    plot!(plt, 
        100*rs/cpt, ws, label=lab, linestyle=lss[i], linewidth=2, color=pal[i]
    )
    scatter!(plt,[100*rs[end]/cpt],[0], label="", color=pal[i],markerstrokewidth=0)
end
plot!(plt,
    # legendtitle="Avail. workers", # impossible to center
    foreground_color_legend = nothing,
    xlabel = "Elapsed-time / CPU-time (%)",
    ylabel = "Number of busy workers",
    size   = (450,225)
)
ys = first(first(yticks(plt)))
yticks!(plt,ys,NRSTExp.fmt_thousands.(ys))
savefig(plt, "runtime.pdf")

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


