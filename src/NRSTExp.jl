module NRSTExp

using Random: AbstractRNG
using DataFrames
using Dates: Dates
using DelimitedFiles: writedlm
using CSV: CSV
using NRST

export dispatch

include("dispatcher.jl")
include("benchmark.jl")

# sub-modules
include("IdealIndexProcesses/IdealIndexProcesses.jl")
using .IdealIndexProcesses

include("ExamplesGallery/ExamplesGallery.jl")
using .ExamplesGallery

end # module
