using Test
using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using NRSTExp.CompetingSamplers
using NRSTExp.IdealIndexProcesses
using LinearAlgebra
using LogExpFunctions
using SplittableRandoms

include("testutils.jl")

@testset "NRSTExp" begin
    include("IdealIndexProcesses.jl")
    include("CompetingSamplers.jl")
end
