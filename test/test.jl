using NRSTExp
pars = Dict(
    "exp"  => "benchmark",    
    "mod"  => "Challenger",
    "cor"  => "0.8",
    "gam"  => "1.0",
    "fun"  => "median",
    "seed" => "125"
)
dispatch(pars)

# julia --project -e "using NRSTExp; dispatch()" exp=benchmark mod=Challenger cor=0.8 gam=1.0 fun=median seed=125