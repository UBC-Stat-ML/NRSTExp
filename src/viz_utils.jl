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
    title_str = alg_name * " (TE ≈ $(round(TE, digits=2)))"
    xlab      = alg_name == "NRST"
    piproc    = plot(
        0:(length(is)-1), is, grid = false, palette = DEF_PAL, ylims = (0,N),
        ylabel = "Level", label = "",
        left_margin = 15px, bottom_margin = 15px,
        size = (675, xlab ? 180 : 171),
        titlefontsize = 11
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
    tm       = MvNormalTM(3,2.,2.)              # Λ ≈ 1
    new_rng  = quote SplittableRandom(3509) end # re-use this call to make both samplers run with the same stream
    ns,_,_   = NRSTSampler(tm, eval(new_rng))
    ntours   = 512
    res_nrst = parallel_run(ns,eval(new_rng),NRST.NRSTTrace(ns),ntours=ntours);
    pnrst    = plot_trace_iproc(res_nrst;alg_name="NRST",kwargs...)
    gt       = GT95Sampler(ns);
    res_gt   = parallel_run(gt,eval(new_rng),NRST.NRSTTrace(gt),ntours=ntours);
    pgt      = plot_trace_iproc(res_gt;alg_name="ST",kwargs...)
    savefig(pnrst,"index_process_nrst.pdf")
    savefig(pgt,"index_process_gt.pdf")
    return
end

# estimate barriers and export as CSV for plotting in R
function get_barrier_df()
    tms  = (Banana(), Funnel(), HierarchicalModel(), MRNATrans(), ThresholdWeibull(), XYModel(8));
    labs = ("Banana", "Funnel", "HierarchicalModel", "MRNATrans", "ThresholdWeibull", "XYModel")
    rng  = SplittableRandom(89440)
    df   = vcat((get_barrier_df(tms[i], rng, labs[i]) for i in eachindex(tms))...)
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

###############################################################################
# runtime plot
###############################################################################

function move_fwd!(pq,t)
    for e in keys(pq)
        pq[e] -= t
    end
    pq
end
function to_sorted_array!(pq,rs,T)
    @assert length(pq) + 2 == length(rs)
    rs[1] = zero(eltype(rs))
    rs[2] = T
    for i in 1:length(pq)
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
    k   = P                          # tours taken so far from ts
    T   = 0.                         # init time
    K   = length(ts)
    while k<K
        t  = last(dequeue_pair!(pq)) # jump to next event
        T += t                       # move wall clock fwd
        move_fwd!(pq,t)              # move time forward in the queue
        k += 1                       # take another tour and put in queue
        enqueue!(pq, k, ts[k])
    end
    rs = to_sorted_array!(pq,T)      # queue is emptied here
    ws = [P;P:-1:0]
    (rs,ws)
end
function fmt_thousands(a::Int)
    #a = 556046
    a_str = string(a)
    s = ""
    for i in 1:length(a_str)
        #i=1
        i>1 && rem(i-1,3) == 0 && (s = "," * s)
        p = length(a_str)-i+1
        s = a_str[p] * s
    end
    s
end
fmt_thousands(a::AbstractFloat) = fmt_thousands(round(Int, a))
function make_runtime_plot(;
    Λ    = 5.,                # tempering barrier
    K    = 10000,             # total number of tours
    Prat = 2. .^ range(-4, 0) # vector of proportions of workers over total number of tours
    )
    
    # simulate
    rng = Xoshiro(1)
    ts  = Λ*randexp(rng, K)
    cpt = sum(ts)
    Pvec= round.(Int, K .* Prat)
    pq  = PriorityQueue{Int64,Float64}()
    res = [simulate_events(pq,ts,P) for P in Pvec]
    
    # plot the results
    lss = reverse!([:solid, :dash, :dot, :dashdot, :dashdotdot])
    plt = plot(palette = :seaborn_colorblind6)
    for (i,p) in enumerate(Pvec)
        #i=1; p = Prat[i]
        rs,ws = res[i]
        plot!(plt, 100*rs/cpt, ws, label=fmt_thousands(p), linestyle=lss[i], linewidth=2)
    end
    plot!(plt,
        legendtitle="Avail. workers", foreground_color_legend = nothing,
        xlabel = "Elapsed-time / CPU-time (%)",
        ylabel = "Number of busy workers",
        size   = (500,250)
    )
    ys = first(first(yticks(plt)))
    yticks!(plt,ys,fmt_thousands.(ys))
    savefig(plt, "runtime.pdf")
end