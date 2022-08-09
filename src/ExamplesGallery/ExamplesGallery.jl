module ExamplesGallery

using DelimitedFiles: readdlm
using Distributions
using DynamicPPL
using FillArrays: fill
using IrrationalConstants: twoπ, log2π
using Lattices: Square, edges
using LinearAlgebra: I
using UnPack: @unpack
using ..NRSTExp: NRSTExp
import NRST: NRST, TemperedModel, TuringTemperedModel, V, Vref
import Base: rand

# Turing
include("Turing/hierarchical_model.jl")
export HierarchicalModel

# Physics
include("Physics/XY_model.jl")
export XYModel

# Testing
include("Testing/mvNormals.jl") # example with multivariate Normals admitting closed form expressions
export MvNormalTM, free_energy, get_scaled_V_dist

# define Turing models at startup so they are available when called inside dispatch
# +info: an issue arises when running julia in HPC environments, where the cache
# of precompiled stuff must be disable to avoid race conditions between multiple
# processes. But no having cache causes problems with @generated functions in
# packages used by DynamicPPL. Particularly, we get a "The applicable method may be too new:"
# error for function "is_propertynames_overloaded" in pkg ConstructionBase. See  
# https://docs.julialang.org/en/v1/manual/methods/#Redefining-Methods
using ConstructionBase
function __init__()
    include(pkgdir(NRSTExp, "src", "ExamplesGallery", "Turing", "hierarchical_model_dppl.jl"))
end

end
