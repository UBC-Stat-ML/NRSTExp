module NRSTExp

using ColorSchemes: seaborn_colorblind, seaborn_colorblind6
using DataFrames
using DataStructures
using DelimitedFiles: writedlm
using CSV: CSV
using Interpolations: gradient1
using NRST
using ParetoSmooth: ParetoSmooth
using Plots
using Plots.PlotMeasures: px
using Random: AbstractRNG, Xoshiro, randexp
using SplittableRandoms: SplittableRandom
using UnPack: @unpack

export dispatch, gen_iproc_plots

include("dispatcher.jl")
include("utils.jl")
include("hyperparams.jl")
include("benchmark.jl")
include("benchmark_own_tuning.jl")
include("viz_utils.jl")

# sub-modules
include("IdealIndexProcesses/IdealIndexProcesses.jl")
using .IdealIndexProcesses

include("ExamplesGallery/ExamplesGallery.jl")
using .ExamplesGallery

include("CompetingSamplers/CompetingSamplers.jl")
using .CompetingSamplers

end # module
