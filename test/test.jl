using NRSTExp

push!(ARGS, "ess_versus_cost")
push!(ARGS, "HierarchicalModel")
push!(ARGS, "0.99")
dispatch()
# julia --project -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99
