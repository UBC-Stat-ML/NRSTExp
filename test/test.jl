using NRSTExp
pars = Dict(
    "exp"  => "benchmark",    
    "mod"  => "Transfection",
    "fun"  => "median",
    "cor"  => "0.8",
    "gam"  => "1.0",
    "seed" => "125"
)
dispatch(pars)

# julia --project -e "using NRSTExp; dispatch()" exp=benchmark mod=Challenger fun=median cor=0.8 gam=1.0 seed=125

using Distributions

d = LogUniform(1.2,3.6)
rand(d)
using DelimitedFiles

a,b = (2,3)
u = rand()
exp(log(a) + u*(log(b) - log(a)))
10^(log10(a) + u*(log10(b) - log10(a)))
isfinite(Inf)