##############################################################################
# Implements the 3-parameter Log-Logistic model and data from Chapter 8 in
#     Cheng, R. (2017). Non-standard parametric statistical inference.
# The model features an unbounded likelihood function
# The full generative model (prior created by me) is
#    a ~ U(0,200)
#    b ~ Inv-Gamma(.1,.1)
#    c ~ U(0.1, 10)
# y|abc~ Log-logistic(y;a,b,c)
##############################################################################

#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct ThresholdLogLogistic{TF, TI} <: AbstractThresholdModel{TF, TI}
    ys::Vector{TF}
    a_prior::Uniform{TF}
    b_prior::InverseGamma{TF}
    c_prior::Uniform{TF}
    lenx::TI
end
function ThresholdLogLogistic()
    ThresholdLogLogistic(
        vec(readdlm(pkgdir(NRSTExp, "data", "SteenStickler0.csv"), ',')),
        Uniform(0, 200),
        InverseGamma(.1,.1),
        Uniform(0.1, 10),
        3
    )
end

# method for the likelihood potential
function NRST.V(tm::ThresholdLogLogistic{TF}, x) where {TF}
    a, b, c = x
    acc = zero(TF)
    for y in tm.ys
        isinf(acc += potloglogistic(y, a, b, c)) && break
    end
    return acc
end

# p(x) = (c/b)(z/b)^{c-1}(1+(z/b)^c)^{-2}
# l(x) = log(c) - log(b) + (c-1)log(z) - (c-1)log(b) - 2log(1+(z/b)^c)
# note that
# (z/b)^c = exp(log((z/b)^c)) = exp(c( log(z) - log(b) ))
# so
# log(1+(z/b)^c) = log1pexp(c(log(z) - log(b)))
# and
# l(x) = log(c) + (c-1)log(z) - clog(b) - 2log1pexp(c(log(z) - log(b)))
# V(x) = clog(b) + 2log1pexp(c(log(z) - log(b))) - log(c) - (c-1)log(z)
function potloglogistic(y::TF,a::TF,b::TF,c::TF) where {TF}
    y <= a && return TF(Inf)
    lb = log(b)
    lc = log(c)
    lz = log(y-a)
    c*lb + 2log1pexp(c*(lz-lb)) - lc - (c-1)lz
end

# # make some plots
# res = parallel_run(ns,rng,NRST.NRSTTrace(ns),TE=TE);
# X = collect(hcat(res.xarray[end]...)')
# using Plots,StatsPlots
# using Plots.PlotMeasures: px
# pcorr = corrplot(
#     X, size= (1200,700), label = 'a':'c',formatter=:plain,
#     left_margin = 20px, bottom_margin = 20px
# )
# savefig(pcorr,"pcorr.png")
# histogram(res.trVs[end],label="V(x)")
# savefig("hist.png")

# y1 = first(tm.ys)
# ds = map(a -> -log(y1-a), X[:,1])
# scatter(ds,-res.trVs[end],xlabel="-log(ymin - a)", ylabel="log-likelihood", label="samples @ beta=1")
# savefig("nlz_versus_loglik.png")