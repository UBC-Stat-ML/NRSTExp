using NRSTExp

push!(ARGS, "ess_versus_cost")
push!(ARGS, "HierarchicalModel")
push!(ARGS, "0.99")
dispatch()
# julia -t 4 --project -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99
# julia --project -e "using NRSTExp" ess_versus_cost HierarchicalModel 0.99
# ./julia -e "using NRSTExp; dispatch()" ess_versus_cost HierarchicalModel 0.99
