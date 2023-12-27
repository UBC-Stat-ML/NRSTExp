function make_sym_rej_mat(rng::AbstractRNG, N::Int, Λ::Real=1)
    br = Λ/N
    @assert br ≤ 1 "N is too low; it should be at least $(ceil(Int,Λ)) to handle Λ=$Λ"
    rs = map(u -> min(1, br * (2u)), rand(rng, N)) # slightly corrupted (but unbiased since E[2u]=1) version of optimal tuning
    u  = one(eltype(rs))
    hcat([rs;u], [u;rs])
end
get_N = NRSTExp.IdealIndexProcesses.get_N
symmetrized_rejections(R::AbstractMatrix) = (R[begin:(end-1),1] .+ R[(begin+1):end,2]) ./ 2
symmetrized_rejections(R::Fill) = R[begin:(end-1),1]
symmetrized_rejections(b::BouncyMC) = symmetrized_rejections(b.R)
sum_odds_rejections(rs::AbstractVector) = sum(r->r/(1-r), rs)
sum_odds_rejections(R::AbstractMatrix) = sum_odds_rejections(symmetrized_rejections(R))
sum_odds_rejections(b::BouncyMC) = sum_odds_rejections(b.R)
roundtrip_probability(b::BouncyMC{NonReversibleBouncy}) = inv(1 + sum_odds_rejections(b))
function roundtrip_probability(b::BouncyMC{ReversibleBouncy})
    inv(2get_N(b) + 2sum_odds_rejections(b))
end
true_TE(b::BouncyMC) = inv(2/roundtrip_probability(b)-1)

get_theoretical_quantities(b::BouncyMC, Λ::Real, rep::Int) =
    DataFrame(
        Lambda = Λ, rev = b isa BouncyMC{ReversibleBouncy}, N = get_N(b), rep = rep,
        type = "theory", rtprob = roundtrip_probability(b), TE = true_TE(b)
    )
get_estimated_quantities(b::BouncyMC, counts::Vector, Λ::Real, rep::Int) =
    DataFrame(
        Lambda = Λ, rev = b isa BouncyMC{ReversibleBouncy}, N = get_N(b), rep = rep,
        type = "estimate", rtprob = mean(>(0),counts), TE = NRSTExp.get_TE(counts)
    )

# runs BouncyMC and computes both theoretical and estimated values for the
# Tour Effectiveness and the probability of hitting the top in a tour
function check_theoretical_formulae(;
    seed   = 1,
    Λ      = 1.0,
    Ns     = 2 .^ (1:8),
    ntours = 2^17,
    n_rep  = 30 
    )
    rng    = SplittableRandom(seed)
    times  = Vector{Int}(undef, ntours)
    counts = similar(times)
    dfs = map(Ns) do N
        dfs = map(1:n_rep) do rep
            R = make_sym_rej_mat(rng, N, Λ)
            mapreduce(vcat,(ReversibleBouncy,NonReversibleBouncy)) do TD
                b = BouncyMC{TD}(R)
                run_tours!(b, rng, times, counts)
                vcat(
                    get_theoretical_quantities(b, Λ, rep),
                    get_estimated_quantities(b, counts, Λ, rep)
                )
            end
        end
        vcat(dfs...)
    end
    vcat(dfs...)
end

# visual check of the output of check_theoretical_formulae
# using StatsPlots
# using Plots.PlotMeasures: px
function plot_check_formulae(res::DataFrame)
    # preprocess
    Λ = first(res.Lambda)
    res[!,"N_str"] = string.(res[!,"N"])

    # common properties
    size   = (650,300)
    xlab   = "Grid size (N)"
    mar    = 15px

    plots = map((false,true)) do flag
        res_rev = filter(:rev => ==(flag),res)

        # roundtrip_probability
        p_rtp = @df res_rev groupedboxplot(
            :N_str, 
            :rtprob, 
            group=:type,
            bar_position = :dodge, 
            size=size,
            # xlab=xlab,
            ylab= flag ? "" : "Probability of reaching top",
            left_margin = mar, bottom_margin = mar,
            title = flag ? "Reversible" : "Non-reversible",
            linecolor = :black # avoid "double-border" in the boxes
        )
        lim = flag ? 0.0 : inv(1+Λ)
        hline!(p_rtp, [lim], linestyle=:dot, label="lim N→∞")

        # Tour effectiveness
        p_TE = @df res_rev groupedboxplot(
            :N_str, 
            :TE, 
            group=:type,
            bar_position = :dodge, 
            size=size,
            xlab=xlab,
            ylab=flag ? "" : "Tour Effectiveness",
            left_margin = mar, bottom_margin = mar,
            linecolor = :black # avoid "double-border" in the boxes
        )
        lim = flag ? 0.0 : inv(1+2Λ)
        hline!(p_TE, [lim], linestyle=:dot, label="lim N→∞")

        p = plot(p_rtp, p_TE, layout=(2,1))
        return p
    end
    plot(plots...,layout=(1,2), size=(700,600))
end
