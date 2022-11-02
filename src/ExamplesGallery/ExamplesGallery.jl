module ExamplesGallery

using DelimitedFiles: readdlm
using Distributions
using DynamicPPL
using FillArrays: Fill
using IrrationalConstants: twoπ, log2π
using Lattices: Square, edges
using LinearAlgebra: I
using LogExpFunctions: logistic, log1pexp
using Turing: BernoulliLogit # TODO: remove once this PR is merged into Distributions.jl: https://github.com/JuliaStats/Distributions.jl/pull/1623
using UnPack: @unpack
using ..NRSTExp: NRSTExp
import NRST: NRST, TemperedModel, TuringTemperedModel, V, Vref
import Base: rand

# Turing
include("Turing/hierarchical_model.jl")
include("Turing/challenger.jl")
export HierarchicalModel
export ChalLogistic

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
export MvNormalTM, free_energy, get_scaled_V_dist

end
