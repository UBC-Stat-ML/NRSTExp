const DEF_PAL = seaborn_colorblind

# extract a given number of steps from a vector of traces corresponding to separate tours
function get_first_nsteps(trvec::Vector{TST}; nsteps::Int, thin::Bool=false) where {T,I,TST<:NRST.AbstractTrace{T,I}}
    thin && (nsteps *= 2)
    is  = Vector{I}(undef, nsteps)
    i0s = I[]                                             # steps at which a tour starts
    l   = zero(I)
    for tr in Iterators.takewhile(_ -> (l<nsteps), trvec) # take a new trace as long as we haven't seen enough steps
        push!(i0s, l)
        for ip in Iterators.take(tr.trIP, nsteps-l)       # never takes more than length(tr.trIP)
            l += 1
            is[l] = first(ip)
        end
    end
    return (thin ? (is[1:2:nsteps], i0s ./= 2) : (is, i0s))
end

# function to create a plot of the trace of the (1st comp) of the index process
function plot_trace_iproc(
    res::NRST.TouringRunResults; 
    alg_name="",
    nsteps=250,
    write_title=true,
    draw_reg_times=true,
    kwargs...
    )
    N         = NRST.get_N(res)
    TE        = last(res.toureff)
    is,ibot   = get_first_nsteps(res.trvec; nsteps, kwargs...)
    title_str = alg_name * " (TE = $(round(TE, digits=2)))"
    xlab      = alg_name == "NRST"
    piproc    = plot(
        0:(length(is)-1), is, grid = false, palette = DEF_PAL, ylims = (0,N),
        ylabel = "Level", label = "",
        left_margin = 15px, bottom_margin = 15px,
        size = (675, xlab ? 180 : 171),
        titlefontsize = 11,
        fontfamily = "Helvetica"
    )
    write_title && title!(piproc, title_str)
    xlab && xlabel!(piproc, "Step")
    hline!(piproc, [N], linestyle = :dot, label="", color = :black)
    draw_reg_times && vline!(piproc, ibot, linestyle = :dot, label="")
    # if top_marks
    #     itop = findall(isequal(N), is)#[1:2:end]
    #     scatter!(piproc, itop, [1.025N], markershape = :dtriangle, label="")
    # end
    return piproc
end

function gen_iproc_plots(;kwargs...)
    # define and tune an NRSTSampler as template
    seed     = 3509
    tm       = MvNormalTM(3,2.,2.)              # Λ ≈ 1
    ns,_,_   = NRSTSampler(tm, SplittableRandom(seed))
    ntours   = 512
    res_nrst = parallel_run(ns,SplittableRandom(seed),NRST.NRSTTrace(ns),ntours=ntours);
    pnrst    = plot_trace_iproc(res_nrst;alg_name="NRST",kwargs...)
    gt       = GT95Sampler(ns);
    res_gt   = parallel_run(gt,SplittableRandom(seed),NRST.NRSTTrace(gt),ntours=ntours);
    pgt      = plot_trace_iproc(res_gt;alg_name="ST",kwargs...)
    savefig(pnrst,"index_process_nrst.pdf")
    savefig(pgt,"index_process_gt.pdf")
end

# animation showing how samples change with temp (for slides)
function make_tempering_gifs()
    rng = SplittableRandom(442016192)

    # Banana
    tm = Banana()
    ns, TE, _ = NRSTSampler(tm,rng)
    res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE)
    X = collect(hcat(res.xarray[end]...)')
    colorgrad = cgrad([viridis[begin], viridis[end]],ns.np.N+1 )
    anim = @animate for (i,xs) in enumerate(res.xarray)
        β = @sprintf("%.2e", ns.np.betas[i])
        X = collect(hcat(xs...)');
        scatter(
            X[:,begin], X[:,end], title = "β = $β", label="",xlabel="x[1]",
            markercolor = colorgrad[i], 
            ylabel="x[2]", xlims=(-10,10), ylims=(-50,50)
        )
    end
    gif(anim, "banana.gif", fps=2)

    # Funnel
    tm = Funnel()
    ns, TE, _ = NRSTSampler(tm,rng)
    res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE);
    X = collect(hcat(res.xarray[end]...)');
    colorgrad = cgrad([viridis[begin], viridis[end]],ns.np.N+1 )
    anim = @animate for (i,xs) in enumerate(res.xarray)
        β = @sprintf("%.2e", ns.np.betas[i])
        X = collect(hcat(xs...)');
        scatter(
            X[:,begin], X[:,end], title = "β = $β", label="",
            xlabel="x[1]", ylabel="x[20]", xlims=(-10,10), ylims=(-50,50),
            markercolor = colorgrad[i], 
        )
    end
    gif(anim, "funnel.gif", fps=2)
end

# models used in the paper
function get_tms_and_labels()
    tms  = (Banana(), Funnel(), HierarchicalModel(), MRNATrans(), ThresholdWeibull(), XYModel(8))
    labs = ("Banana", "Funnel", "HierarchicalModel", "MRNATrans", "ThresholdWeibull", "XYModel")
    tms, labs
end

# estimate barriers and export as CSV for plotting in R
function get_barrier_df()
    tms, labs = get_tms_and_labels()
    rng = SplittableRandom(89440)
    df  = vcat((get_barrier_df(tms[i], rng, labs[i]) for i in eachindex(tms))...)
    CSV.write("barriers.csv", df)
end

# get one barrier df
function get_barrier_df(tm::NRST.TemperedModel, rng::SplittableRandom, lab::AbstractString)
    ns, TE, _ = NRSTSampler(tm, rng)
    res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE)
    Λs  = NRST.get_lambdas(NRST.averej(res))
    DataFrame(mod=lab, beta=ns.np.betas, Lambda=Λs, log_grid=ns.np.log_grid)
end

# utility for creating the Λ plot
function plot_Lambda(f_Λ, bs)
    c1 = DEF_PAL[1]
    c2 = DEF_PAL[2]
    p = plot(
        f_Λ, 0., 1., label = "", legend = :bottomright,
        xlim=(0.,1.), color = c1, grid = false, ylim=(0., f_Λ(bs[end])),
        xlabel = "β", ylabel = "Λ(β)"
    )
    plot!(p, [0.,0.], [0.,0.], label="", color = c2)
    for b in bs[2:end]
        y = f_Λ(b)
        plot!(p, [b,b], [0.,y], label="", color = c2)                  # vertical segments
        plot!(p, [0,b], [y,y], label="", color = c1, linestyle = :dot) # horizontal segments
    end
    p
end

# whole pipeline for Lambda plot
function plot_Lambda(tm::NRST.TemperedModel, rng::SplittableRandom)
    ns, TE, _ = NRSTSampler(tm,rng)
    res = parallel_run(ns, rng, NRST.NRSTTrace(ns), TE=TE)
    @unpack betas, log_grid = ns.np
    f_Λnorm, _, Λs = NRST.gen_lambda_fun(betas, NRST.averej(res), log_grid)
    f_Λ = if log_grid
        β -> Λs[end] * f_Λnorm(NRST.floorlog(β))
    else
        β -> Λs[end] * f_Λnorm(β)
    end
    plot_Lambda(f_Λ, betas)
end

# construct the derivative of the barrier function
# for log_grid we have
# Λ(β) = f(log(β)) => Λ'(β) = f'(log(β))/β
function build_rhoprime(f_Λnorm, Λ, log_grid)
    log_grid || return (β -> Λ*gradient1(f_Λnorm, β))
    function rhoprime(β)
        lβ = NRST.floorlog(β)
        Λ*gradient1(f_Λnorm, lβ)/β
    end
end

# utility to make nice log ticks
function make_log_ticks(lxs::AbstractVector{<:Real}, idealdiv::Int=5)
    lmin, lmax   = extrema(lxs)
    tlmin, tlmax = ceil(Int,lmin), floor(Int,lmax)
    width        = tlmax-tlmin
    if width == 0
        return tlmin:tlmax
    end
    candidates   = 1:width 
    divisors     = candidates[findall([width % c == 0 for c in candidates])]
    bestdiv      = divisors[argmin(abs.(divisors .- idealdiv))] # ideal div implies div+1 actual ticks  
    return tlmin:(width÷bestdiv):tlmax
end

###############################################################################
# simulate run times using a mixture of empirical distribution + GPD tail
###############################################################################

# fit an empirical distribution to data
# https://discourse.julialang.org/t/empirical-distribution-type-for-continuous-variables/5676/15?u=miguelbiron
function EmpiricalDistribution(data::Vector{<:Real})
    sort!(data)                                 # sort the observations
    empirical_cdf = ecdf(data)                  # create empirical cdf
    data_clean = unique(data)                   # remove duplicates to avoid allunique error
    cdf_data = empirical_cdf.(data_clean)       # apply ecdf to data
    pmf_data = vcat(cdf_data[1],diff(cdf_data)) # create pmf from the cdf
    DiscreteNonParametric(data_clean,pmf_data)  # define distribution
end

abstract type EmpiricalMix{TR <: Real} end

function Base.rand(rng::AbstractRNG, em::EmpiricalMix)
    rand(rng) < em.p_thresh ? rand(rng, em.tail_dist) : rand(rng, em.bulk_dist)
end
function Random.rand!(rng::AbstractRNG, xs::Vector{<:Real}, em::EmpiricalMix)
    @inbounds for i in eachindex(xs)
        xs[i] = rand(rng, em)
    end
    return xs
end
function Base.rand(rng::AbstractRNG, em::EmpiricalMix{TR}, n::Int) where {TR}
    rand!(rng, Vector{TR}(undef, n), em)
end
function init_mix(ts::AbstractVector{<:Real}, p_thresh::Real)
    t_thresh  = quantile(ts,1-p_thresh)
    bulk_dist = EmpiricalDistribution(ts[ts .<= t_thresh])
    cent_tail = ts[ts .> t_thresh] .- t_thresh
    return (t_thresh, bulk_dist, cent_tail)
end

# Pareto tail
struct EmpiricalParetoMix{TR, TDB <: Distribution, TDT <: Distribution} <: EmpiricalMix{TR}
    p_thresh::TR
    t_thresh::TR
    bulk_dist::TDB
    tail_dist::TDT
end
function EmpiricalParetoMix(ts::AbstractVector{<:Real}, p_thresh::Real=0.2)
    t_thresh, bulk_dist, cts = init_mix(ts, p_thresh)
    tail_dist = GeneralizedPareto(t_thresh, ParetoSmooth.gpd_fit(cts, 1.0, wip=false, sort_sample=true)...)
    EmpiricalParetoMix(p_thresh, t_thresh, bulk_dist, tail_dist)
end

# Weibull tail
struct EmpiricalWeibullMix{TR, TDB <: Distribution, TDT <: Distribution} <: EmpiricalMix{TR}
    p_thresh::TR
    t_thresh::TR
    bulk_dist::TDB
    tail_dist::TDT
end
function EmpiricalWeibullMix(ts::AbstractVector{<:Real}, p_thresh::Real=0.2)
    t_thresh, bulk_dist, cent_tail = init_mix(ts, p_thresh)
    tail_dist = t_thresh + fit_mle(Weibull, cent_tail)
    EmpiricalParetoMix(p_thresh, t_thresh, bulk_dist, tail_dist)
end

###############################################################################
# runtime plot
# note: if the max tour is in the first batch of tours processed, then it is
# possible that processing with fewer workers will finish at same time as 
# with workers=tours.
###############################################################################

function flip_max!(ts)
    K   = length(ts)
    m,i = findmax(ts)
    if i <= K/2               # swap position of max so that plot looks more natural (see comment above)
        j = K-i+1
        ts[i] = ts[j]
        ts[j] = m
    end
    return (m,i)
end
function simulate_tour_times(rng, K, Λ; flip_max=false)
    ts = Λ*randexp(rng, K)
    flip_max && flip_max!(ts)
    ts
end
# move time fwd in the queue <=> substract fixed t from all queue priorities
# key: this does not change relative order! So we can modify their "setindex!"
# method to avoid the percolating phase. See their original method in the link
# https://github.com/JuliaCollections/DataStructures.jl/blob/d0dd2a012ebd07ff31193b21130109baa50cfe2b/src/priorityqueue.jl#L192
function move_fwd!(pq::PriorityQueue{K,V},t) where {K,V}
    @inbounds for key in keys(pq)
        i        = pq.index[key]
        oldvalue = pq.xs[i].second
        pq.xs[i] = Pair{K,V}(key, oldvalue-t)
    end
    pq
end
function to_sorted_array!(pq,rs,T)
    @assert length(pq) + 2 == length(rs)
    rs[1] = zero(eltype(rs))
    rs[2] = T
    @inbounds for i in 1:length(pq)
        rs[i+2] = T + last(dequeue_pair!(pq))
    end
    rs
end
function to_sorted_array!(pq::PriorityQueue{K,V}, T) where {K,V}
    to_sorted_array!(pq, Vector{V}(undef, length(pq)+2), T)
end
function simulate_events(pq,ts,P)
    @assert isempty(pq)
    @assert P <= length(ts)

    # init the queue
    for i in 1:P
        pq[i] = ts[i]
    end
    k   = P                              # tours taken so far from ts
    T   = 0.                             # init time
    K   = length(ts)
    while k<K
        t  = last(dequeue_pair!(pq))     # jump to next event
        T += t                           # move wall clock fwd
        move_fwd!(pq,t)                  # move time forward in the queue
        k += 1                           # take another tour and put in queue
        @inbounds enqueue!(pq, k, ts[k])
    end
    rs = to_sorted_array!(pq,T)          # queue is emptied here
    ws = [P;P:-1:0]
    (rs,ws)
end
function fmt_thousands(a::Int)
    a_str = string(a)
    len   = length(a_str)
    s     = ""
    for i in 1:len
        i>1 && rem(i-1,3) == 0 && (s = "," * s)
        p = len-i+1
        s = a_str[p] * s
    end
    s
end
fmt_thousands(a::AbstractFloat) = fmt_thousands(round(Int, a))
function plot_busy_workers_over_time(
    ts::Vector{<:Real};
    Pvec = 2 .^ (5:9), # vector of number of workers
    size = (450,225),   # size of the plt
    xlab = "Elapsed time (hours)", # no need to be accurate, plot is just a demostration # "Elapsed-time / CPU-time (%)",
    ylab = "Number of busy workers",
    )
    
    # init
    mts,_ = flip_max!(ts) # find max and possibly flip it to the last half of the sample
    pq  = PriorityQueue{Int64,Float64}()
    res = [simulate_events(pq,ts,P) for P in Pvec]
    
    # plot the results
    lss = reverse!([:solid, :dash, :dot, :dashdot, :dashdotdot])
    pal = seaborn_colorblind6
    plt = plot(
        fontfamily = "Helvetica",
        # legendtitle="Avail. workers", # impossible to center
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        xlabel = xlab,
        ylabel = ylab,
        size   = size
    )
    for (i,p) in enumerate(Pvec)
        #i=1; p = Pvec[i]
        rs,ws = res[i]
        lab   = fmt_thousands(p)
        plot!(plt, 
            rs, ws, label=lab, linestyle=lss[i], linewidth=2, color=pal[i]
        )
        scatter!(plt,[rs[end]],[0], label="", color=pal[i],markerstrokewidth=0)
    end
    scatter!(
        plt, [mts], [0], markershape = :xcross, markersize=3, 
        markerstrokewidth=3, color = :black, label = ""
    )
    ys = first(first(yticks(plt)))
    yticks!(plt,ys,fmt_thousands.(ys))
    plt
end

function workers_time_cost_analysis(;
    ntours = 2^11,
    nreps  = 30,
    Pvec   = 2 .^ (0:11)
    # size   = (450,225)
    )
    # build sampler, run tours and get times in milliseconds
    tm  = ChalLogistic()
    rng = SplittableRandom(24576)
    ns  = first(NRSTSampler(tm, rng));
    ts  = 1000 * NRST.get_time.(parallel_run(ns,rng,ntours=ntours).trvec);

    # store times, compute schedules, and save them
    open("raw_times.tsv", "w") do io
        writedlm(io, ts)
    end
    flip_max!(ts) # find max and possibly flip it to the last half of the sample, so that plot looks better
    pq  = PriorityQueue{Int64,Float64}()
    dfp = vcat((
        begin
            rs,ws = simulate_events(pq,ts,P)
            DataFrame(nw = P, rs = rs, ws = ws)
        end
        for P in Pvec)...
    )
    CSV.write("busy_workers_over_time.csv", dfp)

    # # draw histogram and plot of workers versus time
    # ph  = make_tour_times_hist(ts,size=size)
    # pbw = plot_busy_workers_over_time(ts,size=size)
    # plt = plot(
    #     ph, pbw, size=(2*size[1],size[2]), bottom_margin=20px, right_margin=15px,
    #     left_margin=15px
    # )
    # savefig(plt, "tours_busy_workers.pdf")

    # build and save dataframe with simulations of elapsed time and costs
    ewm = EmpiricalWeibullMix(ts);
    pq  = PriorityQueue{Int64,Float64}()
    res = [begin
            ts = rand(rng, ewm, ntours)
            sts= sum(ts)
            [begin
              rs,_ = simulate_events(pq,ts,P)
              (nw = P, rep = r, et = last(rs), chpc = P*last(rs), clam = sts)
            end for P in Pvec]
        end for r in 1:nreps];
    CSV.write("workers_time_cost.csv", DataFrame(collect(Base.Flatten(res))))
end

function make_tour_times_hist(ts::AbstractVector;
    trunc_prob = 0.025,
    size = (450,225)
    )
    t_trunc = quantile(ts,1-trunc_prob)
    trunc_ts= map(Base.Fix1(min,t_trunc), ts)
    ph = histogram(
        trunc_ts,
        normalize  = :probability,
        fontfamily = "Helvetica",
        label      = "",
        xlabel     = "CPU time of a tour (hours)",
        ylabel     = "Probability",
        size       = size,
        linecolor  = :match
    )
    last_xlab   = last(last(first(xticks(ph))))
    t_trunc_str = ">$(round(t_trunc, digits=length(last_xlab)-2))"
    annotate!(ph, t_trunc, 1.8*trunc_prob, text(t_trunc_str, pointsize=8, family="Helvetica"))
    ph
end

