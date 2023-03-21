# Define a `TemperedModel` type and implement `NRST.V`, `NRST.Vref`, and `Base.rand` 
struct MvNormalTM{TI<:Int,TF<:AbstractFloat} <: TemperedModel
    d::TI
    m::TF
    s0::TF
    s0sq::TF
end
MvNormalTM(d,m,s0) = MvNormalTM(d,m,s0,s0*s0)
NRST.V(tm::MvNormalTM, x) = 0.5sum(xi -> abs2(xi - tm.m), x)  # 0 allocs, versus "x .- m" which allocates a temp
NRST.Vref(tm::MvNormalTM, x) = 0.5sum(abs2,x)/tm.s0sq
Random.rand!(tm::MvNormalTM, rng, x) = map!(_ -> tm.s0*randn(rng), x, x)
Base.rand(tm::MvNormalTM, rng) = tm.s0 * randn(rng, tm.d)

# Write methods for the analytical expressions for ``\mu_b``, 
# ``s_b^2``, and ``\mathcal{F}``
sbsq(tm,b) = inv(inv(tm.s0sq) + b) # use sbsq(tm,b) * I to get a diag matrix
mu(tm,b)   = b * tm.m * sbsq(tm,b) # use it with Fill() to get a lazy vector
function free_energy(tm::MvNormalTM,b::Real)
    m   = tm.m
    ssq = sbsq(tm, b)
    -0.5*tm.d*( log2π + log(ssq) - b*m*m*(1-b*ssq) )
end
free_energy(tm::MvNormalTM, bs::AbstractVector{<:Real}) = map(b->free_energy(tm,b), bs)

# Distribution of the potential function
function get_V_dist(tm,b)
    s² = sbsq(tm,b)
    s  = sqrt(s²)
    μ  = tm.m*(b*s²-1)/s
    scd= NoncentralChisq(tm.d,tm.d*μ*μ)
    (s²/2)*scd
end
