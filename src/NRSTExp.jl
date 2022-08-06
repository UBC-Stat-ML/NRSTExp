module NRSTExp

using Random: AbstractRNG
using DataFrames
using Dates: Dates
using DelimitedFiles: writedlm
using CSV: CSV
using NRST

include("dispatcher.jl")
export dispatch
include("ESSversusCost.jl")
export ess_versus_cost

# sub-modules
include("IdealIndexProcesses/IdealIndexProcesses.jl")
using .IdealIndexProcesses

include("ExamplesGallery/ExamplesGallery.jl")
using .ExamplesGallery

end # module
