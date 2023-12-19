using DataFrames
using FillArrays
using LinearAlgebra
using LogExpFunctions
using Random
using SplittableRandoms
using Statistics
using Test

using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using NRSTExp.CompetingSamplers
using NRSTExp.IdealIndexProcesses

@testset "NRSTExp" begin
    include("IdealIndexProcesses.jl")
    include("CompetingSamplers.jl")
end
