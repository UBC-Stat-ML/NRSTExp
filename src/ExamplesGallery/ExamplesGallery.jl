module ExamplesGallery

using DelimitedFiles: readdlm
using Distributions
using DynamicPPL
using FillArrays: Fill
using IrrationalConstants: twoπ, log2π, logtwo
using Lattices: Square, edges
using LinearAlgebra: I, Diagonal, dot
using LogExpFunctions: logistic, log1pexp
using Random: AbstractRNG
using Turing: filldist, BernoulliLogit
using UnPack: @unpack
using ..NRSTExp: NRSTExp
import NRST: NRST, TemperedModel, TuringTemperedModel, V, Vref
import Base: rand

# utils
include("utils.jl")
export HalfCauchy

# Turing
include("Turing/hierarchical_model.jl")
include("Turing/challenger.jl")
include("Turing/MRNATransfection.jl")
include("Turing/Titanic.jl")
export HierarchicalModel, HierarchicalModelTuring
export ChalLogistic, ChalLogisticTuring
export MRNATrans, MRNATransTuring
export TitanicTuring

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
include("Testing/HalfCauchyEnergy.jl")
export MvNormalTM, free_energy, get_scaled_V_dist
export HalfCauchyEnergy

end
