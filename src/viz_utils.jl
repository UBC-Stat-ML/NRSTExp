const DEF_PAL = seaborn_colorblind

# extract a given number of steps from a vector of traces corresponding to separate tours
function get_first_nsteps(trvec::Vector{TST}, nsteps) where {T,I,TST<:NRST.AbstractTrace{T,I}}
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
    return is,i0s
end

# function to create a plot of the trace of the (1st comp) of the index process
function plot_trace_iproc(res::NRST.TouringRunResults; nsteps=250,xlab=true)
    N       = NRST.get_N(res)
    is,ibot = get_first_nsteps(res.trvec, nsteps)
    piproc  = plot(
        0:(length(is)-1), is, grid = false, palette = DEF_PAL, ylims = (0,N),
        ylabel = "Level", label = "",
        left_margin = 15px, bottom_margin = 15px,
        size = (675, xlab ? 180 : 171)
    )
    xlab && xlabel!(piproc, "Step")
    hline!(piproc, [N], linestyle = :dot, label="", color = :black)
    vline!(piproc, ibot, linestyle = :dot, label="")
    # if top_marks
    #     itop = findall(isequal(N), is)#[1:2:end]
    #     scatter!(piproc, itop, [1.025N], markershape = :dtriangle, label="")
    # end
    return piproc
end

function gen_iproc_plots()
    # define and tune an NRSTSampler as template
    tm       = MvNormalTM(3,2.,2.)              # Λ ≈ 1
    new_rng  = quote SplittableRandom(3509) end # re-use this call to make both samplers run with the same stream
    ns, TE,Λ = NRSTSampler(tm, eval(new_rng))
    res_nrst = parallel_run(ns,eval(new_rng),NRST.NRSTTrace(ns),ntours=50);
    pnrst    = plot_trace_iproc(res_nrst)
    gt       = GT95Sampler(ns);
    res_gt   = parallel_run(gt,eval(new_rng),NRST.NRSTTrace(gt),ntours=50);
    pgt      = plot_trace_iproc(res_gt,xlab=false)
    savefig(pnrst,"index_process_nrst.pdf")
    savefig(pgt,"index_process_gt.pdf")
    return
end