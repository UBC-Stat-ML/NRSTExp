##############################################################################
# simple logistic regression with QR decomposition
#     σ   ~ Exp(1)     
#     α   ~ Cauchy(0,σ)
#     β_j ~ Cauchy(0,σ) iid
#     y_i ~ Bernoulli(logistic(α + <β,x_i>))
# IDEA 1: center X: the centering matrix is
#   C = 1m^T
# where m := 1^TX/n is the mean of each column. Then
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
#   α1 + Xβ = (α + m^Tβ(θ))1 + Qθ = (α + m^T * Rinv * θ)1 + Qθ
##############################################################################

#######################################
# pure julia version
#######################################

# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct Titanic{TF<:AbstractFloat, TI<:Int} <: TemperedModel
    Q::Matrix{TF}             # scaled Q in the QR decomposition of X
    Rinv::Matrix{TF}          # inverse of scaled R
    Rrows::Vector{Vector{TF}} # rows of R, used in forward simulation β -> θ
    Ritm::Vector{TF}          # Rinv' * m, where m is the vector of column means of X
    y::BitVector              # labels
    Qθ::Vector{TF}            # temp storage
    β::Vector{TF}             # temp storage
    n::TI
    p::TI
    lenx::TI
    E::Exponential{TF}
end
function Titanic()
    X, y  = titanic_load_data()
    n, p  = size(X)
    m     = mean(X,dims=1)                # colmeans as 1xp matrix
    X   .-= m                             # center in place (m is "virtually broadcasted" to nxp)
    Qf,R  = qr(X)                         # decompose Xc = X-m into Q (full) and R
    s     = sqrt(n-1)                     # scale factor
    Q     = Matrix(Qf)*s                  # thin and scale to ensure Cov(Q)==I
    R   ./= s                             # apply inverse scale factor in place
    Rinv  = inv(R)                        # invert
    Rrows = [copy(r) for r in eachrow(R)] # split R into rows for easier forward simulation of β -> θ(β)
    Titanic(
        Q, Rinv, Rrows, Rinv' * vec(m), y, similar(X,n), similar(X,p),
        n, p, 2+p, Exponential()
    )
end

# copy method. keeps everything common except temp storages Qθ and β. 
# this is needed in order to avoid race conditions when sampling in parallel
function Base.copy(tm::Titanic)
    Titanic(
        tm.Q, tm.Rinv, tm.Rrows, tm.Ritm, tm.y, similar(tm.Qθ), similar(tm.β),
        tm.n, tm.p, tm.lenx, tm.E
    )
end

# split x into components
function split(tm::Titanic{TF}, x::AbstractVector{TF}) where {TF}
    ( lσ = x[1], α = x[2], θ = view(x, 3:tm.lenx) )
end

# prior potential. note: this is missing the additive const c=-logabsdet(Rinv)
function NRST.Vref(tm::Titanic{TF}, x) where {TF}
    lσ, α, θ   = split(tm, x)
    σ          = exp(lσ)
    (σ < eps(TF)) && return TF(Inf)
    acc        = zero(TF)
    isinf(acc -= logpdf(tm.E, σ)) && return acc
    acc       -= lσ                               # logabsdetjac
    C          = Cauchy(zero(TF), σ)
    isinf(acc -= logpdf(C, α)) && return acc
    mul!(tm.β, tm.Rinv, θ)
    for βᵢ in tm.β
        isinf(acc -= logpdf(C, βᵢ)) && return acc
    end
    return acc
end
function Random.rand!(tm::Titanic{TF}, rng, x) where {TF}
    σ    = rand(rng, tm.E)
    x[1] = log(σ)
    C    = Cauchy(zero(TF), σ)
    x[2] = rand(rng, C)
    rand!(rng, C, tm.β)                           # sample β, store in temp storage
    for i in 3:tm.lenx
        @inbounds x[i] = dot(tm.Rrows[i-2], tm.β) # get θ[j]=R[j,:]*β
    end
    return x
end
function Base.rand(tm::Titanic{TF}, rng) where {TF}
    rand!(tm, rng, Vector{TF}(undef, tm.lenx))
end

# method for the likelihood potential
function NRST.V(tm::Titanic{TF}, x) where {TF}
    _, α, θ = split(tm, x)
    mul!(tm.Qθ, tm.Q, θ)
    α_c = dot(tm.Ritm, θ)                      # == m^T * Rinv * θ
    acc = zero(TF)
    for (i, yᵢ) in enumerate(tm.y)
        @inbounds ℓ = α + α_c + tm.Qθ[i]       # == α+Xβ. 2.5 times faster with @inbounds
        acc += yᵢ ? log1pexp(-ℓ) : log1pexp(ℓ) # log1pexp is never Inf if ℓ isn't, so no need for a check here
    end
    return acc
end

# function titanic_load_data()
#     dta = readdlm(pkgdir(NRSTExp, "data", "titanic.csv"), ',', skipstart=1)
#     y   = dta[:,1] .> 0
#     X   = dta[:,2:end]
#     return X, y
# end

# # use the NoQR version together with this identity
# #   1 = E_p[1] = E_p[q/q] = E_q[p/q] = E_q[exp(c-V_p+V_q)]
# # to check that adding c would give a normalized Vref for the QR version
# 
# using LinearAlgebra, Random
# tm  = Titanic();
# tm2 = TitanicNoQR();
# rng = SplittableRandom(7152)
# c = first(logabsdet(tm.Rinv))
# x = similar(tm.β,tm.lenx)
# rand!(tm,rng,x)
# NRST.V(tm,x) ≈ NRST.V(tm2,[x[1:2];tm.Rinv*x[3:end]]) # check the bijection leaves V unchanged
# S = 1_000_000_000
# lS=log(S)
# sum(_ -> (rand!(tm2,rng,x); exp(c-NRST.Vref(tm,x)+NRST.Vref(tm2,x)-lS)), 1:S ) # =0.7345778272753167 --> good enough? :/
