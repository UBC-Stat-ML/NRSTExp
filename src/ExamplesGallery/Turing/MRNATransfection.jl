#######################################
# pure julia version
#######################################



function mrna_trans_load_data()
    dta = readdlm(pkgdir(NRSTExp, "data", "transfection.csv"), ',')
    ts  = Float64.(dta[2:end,1])
    ys  = Float64.(dta[2:end,3])
    return ts, ys
end

#######################################
# turing version
#######################################

# Define a model using the `DynamicPPL.@model` macro.
# x = exp(log(a) + u(log(b) - log(a)))
# = exp(log(10)[log(a) + u(log(b) - log(a))]/log(10) )
# = 10^[log10(a) + u(log10(b) - log10(a))]
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
