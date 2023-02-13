using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(2798)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean   = false,
    maxcor     = 0.95,
    γ          = 10.0,
    xpl_smooth_λ = 0.1,
);
ntours = ceil(Int,16.779857306264404*4096)
res=parallel_run(ns,rng,ntours=ntours);
res.visits[end,1]
TE=res.toureff[end]
res=parallel_run(ns,rng,TE=TE);

# IDEA 1: center X! The centering matrix is
#   C = 1m^T
# where 
#   m := 1^TX/n is the mean of each column
# Then
#   α1 + Xβ = α1 + [(X-C)+C]β = (α1 + Cβ) + Xcβ
# But 
#   Cβ = (1m^T)β = 1(m^Tβ)
# Hence
#   α1 + Xβ = (α + m^Tβ)1 + Xcβ
# IDEA 2: use thin Xc=QR decomposition and scale it to get I second moment
#   Q0,R0 = qr(Xc)
#   Q = Q0*sqrt(n-1), R = R0/sqrt(n-1) ---> scalar cancels so QR=Q0R0=Xc
# then, exploit
#   Xcβ = Q0R0β = QRβ = Qθ
# where 
#   θ := Rβ <=> β = R \ θ = inv(R) * θ
# note: this is a linear transformation, so the Jacobian is constant
#   p(θ) = q(β(θ)) det[Jac[β(θ)]] ∝ q(β(θ)) = ∏_j Cauchy(β_j(θ))
# but, to evaluate Vref, we then need to compute β(θ) = inv(R) * θ
# BUT: @btime ($Rinv * $θ) ~ 90 ns ~ 0.5% cost of Qθ (1645 ns), so no biggie!
# Finally, the linear predictor in the (α,θ) coordinates becomes
#   α1 + Xβ = (α + m^Tβ(θ))1 + Qθ
using LinearAlgebra, Statistics, Random, Distributions
n,d=size(tm.X)
m = mean(tm.X, dims=1)
Xc = tm.X .- m
Q0,R0 = qr(Xc);
norm(Q0' * Q0 - I)
Q = Matrix(Q0)*sqrt(n-1) # thin and scale
R = R0/sqrt(n-1)
norm(mean(Q, dims=1))    # Q is also 0 mean
norm((Q' * Q)/(n-1) - I) # 2nd moment matrix == I
norm(cov(Q) - I)         # cov matrix == I (since mean of Q is 0)
α= 13.
β = randn(d)
rand!(MersenneTwister(),Cauchy(0,3),β)
θ = R*β
R_rows = [copy(r) for r in eachrow(R)]
dot(R_rows[begin],β)
x = similar(θ,d+2)
mul!(view(x,3:(d+2)), R*β)
Rinv = inv(R)
β = Rinv*θ
norm(dot(m,β) .+ Q*θ .- tm.X*β)
dot(m, Rinv, θ)

using BenchmarkTools

@btime ($Qt * $θ)
Xc = tm.X .- mean(tm.X,dims=1)
A = copy(tm.X)
A .-= mean(A,dims=1)
Xc == A
ones(n)*(ones(n)' * tm.X)/n

###############################################################################
# example system call
###############################################################################

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=Challenger  \
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=1111

# julia --project -t 4 \
#     -e "using NRSTExp; dispatch()" \
#     exp=hyperparams  \
#     mod=HierarchicalModel  \
#     fun=median    \
#     cor=0.95 \
#     gam=10.0  \
#     xps=0.1 \
#     seed=2798

###############################################################################
# end
###############################################################################
