using NRSTExp
using NRSTExp.IdealIndexProcesses
using Test

@testset "NRSTExp" begin
    @testset "IdealIndexProcesses" begin
        @testset "BouncyMC" begin
            N = 11
            tourls, vNs = run_tours!(BouncyMC(0.,N), 5)
            @test all(tourls .== 2N+2)
            @test all(vNs .== 1)
        end
    end
end