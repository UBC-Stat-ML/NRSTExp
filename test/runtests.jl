include("supporting/setup.jl")

@testset "NRSTExp" begin
    include("IdealIndexProcesses.jl")
    include("CompetingSamplers.jl")
end
