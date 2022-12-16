@testset "CompetingSamplers" begin
    # define and tune an NRSTSampler as template
    tm  = ChalLogistic()
    rng = SplittableRandom(5470)
    ns, TE, Λ = NRSTSampler(
        tm,
        rng,
        N=12,
        adapt_N_rounds=0
    )

    # set V to a value of the finite differences of c
    N     = ns.np.N
    nlvls = N+1
    ri    = ceil(Int, rand(rng)*N)
    dc    = (ns.np.c[ri+1]-ns.np.c[ri]) / (ns.np.betas[ri+1]-ns.np.betas[ri])
    ns.curV[] = dc

    # instantiate samplers
    sh   = SH16Sampler(ns);
    fbdr = FBDRSampler(ns);
    NRSTExp.CompetingSamplers.update_gs!(fbdr);
    π∞_tru = exp.(fbdr.gs); # actual target == conditional dist of beta given V

    @testset "SH16Sampler" begin
        P = buildTransMat(sh)
        @test all(sum(P,dims=2) .≈ 1.)

        # get stationary distribuion <=> get left-nullspace of P-I 
        # <=> get right-nullspace of P'-I
        π∞ = nullspace(P'-I)[:,1]
        π∞ = π∞ / sum(π∞)
        @test all(π∞ .>= -eps())                 # <=1 implicit by imposing sum()=1
        @test π∞[1:nlvls] ≈ π∞[(nlvls+1):2nlvls] # i is indep of eps under stationary dist
        @test 2π∞[1:nlvls] ≈ exp.(fbdr.gs)       # times 2 <=> marginalize eps

        # check basic SDBC properties
        T⁺, T⁻, Λ⁺, Λ⁻ = splitTransMat(P)
        @test Λ⁺ - Λ⁻ ≈ sum(T⁻ - T⁺,dims=2)      # this is implied
        @test diag(T⁺) ≈ diag(T⁻)                # this condition is not explicit in SDBC papers but it's true
    end

    @testset "FBDRSampler" begin
        P = buildTransMat(fbdr)
        @test all(sum(P,dims=2) .≈ 1.)

        # get stationary distribuion <=> get left-nullspace of P-I 
        # <=> get right-nullspace of P'-I
        π∞ = nullspace(P'-I)[:,1]
        π∞ = π∞ / sum(π∞)
        @test all(π∞ .>= -eps())                 # <=1 implicit by imposing sum()=1
        @test π∞[1:nlvls] ≈ π∞[(nlvls+1):2nlvls] # i is indep of eps under stationary dist
        @test 2π∞[1:nlvls] ≈ exp.(fbdr.gs)       # times 2 <=> marginalize eps

        # check basic SDBC properties
        T⁺, T⁻, Λ⁺, Λ⁻ = splitTransMat(P)
        @test Λ⁺ - Λ⁻ ≈ sum(T⁻ - T⁺,dims=2)      # this is implied
        @test diag(T⁺) ≈ diag(T⁻)                # this condition is not explicit in SDBC papers but it's true

        # check formula for staying probs holds
        # need to build the transition matrix of the original (reversible) Metropolized Gibbs
        M = collect(hcat(map(idx -> MetroGibbs(fbdr.gs,idx),1:nlvls)...)')
        π∞_M = nullspace(M'-I)[:,1]
        π∞_M = π∞_M ./ sum(π∞_M)
        @test π∞_M ≈ exp.(fbdr.gs)               # correct stationary distribution
        @test 2diag(T⁻) ≈ 1 .+ diag(M)-(Λ⁺ + Λ⁻) # formula for prob of staying (not explicit in the papers either)
    end
end
