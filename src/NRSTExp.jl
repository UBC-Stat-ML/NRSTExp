module NRSTExp

using Random: AbstractRNG
using DataFrames
using DelimitedFiles: writedlm
using CSV: CSV
using NRST
using ParetoSmooth: ParetoSmooth
using SplittableRandoms: SplittableRandom
using Plots
using Plots.PlotMeasures: px
using ColorSchemes: seaborn_colorblind

export dispatch, gen_iproc_plots

include("dispatcher.jl")
include("utils.jl")
include("hyperparams.jl")
include("benchmark.jl")
include("viz_utils.jl")

# sub-modules
include("IdealIndexProcesses/IdealIndexProcesses.jl")
using .IdealIndexProcesses

include("ExamplesGallery/ExamplesGallery.jl")
using .ExamplesGallery

include("CompetingSamplers/CompetingSamplers.jl")
using .CompetingSamplers

end # module
