#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct MRNATrans{TF<:AbstractFloat} <: TemperedModel
    as::Vector{TF}
    bs::Vector{TF}
    bma::Vector{TF} # b-a
    ts::Vector{TF}
    ys::Vector{TF}
end
MRNATrans(as,bs,ts,ys) = MRNATrans(as,bs,bs .- as,ts,ys)
function MRNATrans()
    MRNATrans(
        [-2., -5., -5., -5., -2.],
        [ 1.,  5.,  5.,  5.,  2.],
        mrna_trans_load_data()...
    )
end

function mrna_trans_load_data()
    dta = readdlm(pkgdir(NRSTExp, "data", "transfection.csv"), ',', skipstart=1)
    ts  = dta[:,1]
    ys  = dta[:,3]
    return ts, ys
end

# methods for the prior
# if x~U[a,b] and y = f(x) = 10^x = e^{log(10)x}, then
# P(Y<=y) = P(10^X <= y) = P(X <= log10(y)) = F_X(log(y))
# p_Y(y) = d/dy P(Y<=y) = p_X(log(y)) 1/y = [ind{10^a<= y <=10^b}/(b-a)] [1/y]
# which is the reciprocal distribution or logUniform, so it checks out
function NRST.Vref(tm::MRNATrans{TF}, x) where {TF}
    vr = zero(TF)
    for (i,x) in enumerate(x)
        if x < tm.as[i] || x > tm.bs[i] 
            vr = TF(Inf)
            break
        end
    end
    return vr
end
function Random.rand!(tm::MRNATrans, rng, x)
    for (i, a) in enumerate(tm.as)
        x[i] = a + rand(rng) * tm.bma[i]
    end
    return x
end
function Base.rand(tm::MRNATrans{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, length(tm.as)))
end

# method for the likelihood potential
function NRST.V(tm::MRNATrans{TF}, x) where {TF}
    t₀  = 10^x[1]
    km₀ = 10^x[2]
    β   = 10^x[3]
    δ   = 10^x[4]
    σ   = 10^x[5]
    κ   = (km₀ /(δ - β))
    acc = zero(TF)
    for (n, t) in enumerate(tm.ts)
        tmt₀ = t - t₀
        @inbounds y = tm.ys[n]
        μ = κ * expamexpb(-β*tmt₀, -δ*tmt₀)
        isnan(μ) && error("μ=$μ => t₀=$t₀, km₀=$km₀, β=$β, δ=$δ, -β*tmt₀=$(-β*tmt₀), -δ*tmt₀=$(-δ*tmt₀), x=$x")
        isinf(acc -= logpdf(Normal(μ, σ), y)) && break
    end
    return acc
end

# exp(a) - exp(b) = exp(max(a,b))[exp(a-max(a,b)) - exp(b-max(a,b))]
# = 1{a>b}{exp(a)[1 - exp(b-a)]} + 1{a<=b}{exp(b)[exp(a-b) - 1]}
# = 1{a>b}{-exp(a)expm1(b-a)} + 1{a<=b}{exp(b)expm1(a-b)}
expamexpb(a,b) = a>b ? -exp(a)*expm1(b-a) : exp(b)*expm1(a-b)

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
@model function _mrna_trans(ts,ys)
    t0    ~ LogUniform(1e-2, 1e1)
    km0   ~ LogUniform(1e-5, 1e5)
    beta  ~ LogUniform(1e-5, 1e5)
    delta ~ LogUniform(1e-5, 1e5)
    sigma ~ LogUniform(1e-2, 1e2)
    for n in eachindex(ts)
        t = ts[n]
        m = km0 / (delta - beta) * (-expm1(-(delta - beta) * (t - t0))) * exp(-beta*(t - t0))
        !isfinite(m) && (m = 1e4)
        ys[n] ~ Normal(m,sigma)
    end
end

# Loading the data and instantiating the model
function MRNATransTuring()
    ts, ys = mrna_trans_load_data()
    model  = _mrna_trans(ts,ys)
    return TuringTemperedModel(model)
end
