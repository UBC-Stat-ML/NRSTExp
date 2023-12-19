include("supporting/testutils_IdealIndexProcesses.jl")

@testset "IdealIndexProcesses" begin
    rng = SplittableRandom(1)
    
    @testset "BouncyMC" begin
        @testset "changes in y match direction" begin
            n_steps = 10
            for TD in (NonReversibleBouncy, ReversibleBouncy)
                b = BouncyMC{TD}(0.5, 4)
                NRST.renew!(b, rng)
                y,e = b.state
                for _ in 1:n_steps
                    NRST.step!(b, rng)
                    y_new, e_new = b.state
                    @test y_new == y || y_new-y == e_new
                    y = y_new
                end
            end
        end
        @testset "NonReversibleBouncy" begin
            N = 16

            # deterministic at rho=0
            tourls, vNs = run_tours!(BouncyMC(0.,N), rng; ntours=512)
            @test all(==(2N+2), tourls)
            @test all(==(2), vNs)

            # deterministic at rho=1
            tourls, vNs = run_tours!(BouncyMC(1.,N), rng; ntours=512)
            @test all(==(2), tourls)
            @test all(==(0), vNs)

            # is a periodic Markov chain
            tourls, vNs = run_tours!(BouncyMC(.5,N), rng; ntours=512)
            @test all(iseven, tourls)
            @test all(iseven, vNs)
        end
        @testset "ReversibleBouncy" begin
            # is not a periodic Markov chain
            N = 16
            tourls, vNs = run_tours!(BouncyMC{ReversibleBouncy}(.5,N), rng; ntours=512)
            @test !all(iseven, tourls)
            @test !all(iseven, vNs)
            @test !all(isodd, tourls)
            @test !all(isodd, vNs)
        end
        @testset "Theoretical formulae" begin
            thresh = 0.01
            res = check_theoretical_formulae()
            @test all(d -> abs(d) < thresh, res[2:2:end,:TE] .- res[1:2:end,:TE])
            @test all(d -> abs(d) < thresh, res[2:2:end,:rtprob] .- res[1:2:end,:rtprob])
        end
    end
end
