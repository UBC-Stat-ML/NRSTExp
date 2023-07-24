module ExamplesGallery

using DelimitedFiles: readdlm
using Distributions
# using DistributionsAD: filldist
using DynamicPPL
using FillArrays: Fill
using IrrationalConstants: twoπ, log2π, logtwo
using Lattices: Square, edges
using LazyArrays
using LinearAlgebra
using LogExpFunctions: logistic, log1pexp, logsumexp
using Random
using Statistics: mean
using UnPack: @unpack
using ..NRSTExp: NRSTExp
import NRST: NRST, TemperedModel, TuringTemperedModel, V, Vref
import Base: rand
import Random: rand!

# utils
include("utils.jl")
export HalfCauchy

# Turing
include("Turing/hierarchical_model.jl")
include("Turing/challenger.jl")
# include("Turing/galaxy.jl")
include("Turing/MRNATransfection.jl")
# include("Turing/Titanic.jl")
# include("Turing/Titanic_no_QR.jl")
include("Turing/TitanicHS.jl")
include("Turing/ThresholdWeibull.jl")
include("Turing/ThresholdLogLogistic.jl")
export HierarchicalModel, HierarchicalModelTuring
export ChalLogistic, ChalLogisticTuring
# export GalaxyTuring
export MRNATrans, MRNATransTuring
export Titanic, TitanicNoQR, TitanicHS
export ThresholdWeibull, ThresholdLogLogistic

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
include("Testing/HalfCauchyEnergy.jl")
include("Testing/Funnel.jl")
include("Testing/Banana.jl")
export MvNormalTM, free_energy, get_scaled_V_dist
export HalfCauchyEnergy
export Funnel
export Banana

end
