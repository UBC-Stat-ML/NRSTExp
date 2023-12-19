###############################################################################
# regenerative simulation of the finite state index process
###############################################################################

# type of index process dynamics
abstract type BouncyDynamics end
struct ReversibleBouncy <: BouncyDynamics end
struct NonReversibleBouncy <: BouncyDynamics end

# define a type
struct BouncyMC{TD<:BouncyDynamics, TI<:Int, TM<:AbstractMatrix{<:AbstractFloat}} <: Bouncy
    R::TM                # (N+1)×2 matrix of rejection probs. R[:,1] is up, R[:,2] is dn, R[1,2]=R[N+1,1]=1.0
    state::MVector{2,TI} # current state (position, direction)
end

# constructors
function BouncyMC{TD}(R::AbstractMatrix) where {TD<:BouncyDynamics}
    TI = typeof(size(R,1))
    BouncyMC{TD,TI,typeof(R)}(R, MVector{2,TI}(undef))
end
function BouncyMC{TD}(ρ::AbstractFloat, N::Int) where {TD<:BouncyDynamics}
    BouncyMC{TD}(Fill(ρ, (N+1, 2)))
end
BouncyMC(args...) = BouncyMC{NonReversibleBouncy}(args...) # nonreversible by default
get_N(b::BouncyMC) = size(b.R,1)-1

# step method
# only change between dynamics is how the direction is handled
get_direction(b::BouncyMC{NonReversibleBouncy}, _) = last(b.state)
function get_direction(::BouncyMC{ReversibleBouncy,TI}, rng::AbstractRNG) where {TI} 
    rand(rng, (-one(TI), one(TI)))
end
function step!(bouncy::BouncyMC, rng::AbstractRNG)
    y = first(bouncy.state)
    e = get_direction(bouncy,rng)                 # get a direction
    next_y = y + e                                # proposed state
    if next_y > get_N(bouncy)                     # bounce above
        e = -one(e)
    elseif next_y < zero(y)                       # bounce below
        e = one(e)
    elseif rand(rng) < bouncy.R[y+1, e>0 ? 1 : 2] # reject move?
        e = -e
    else                                          # accept move
        y = next_y
    end

    # save state
    @inbounds begin
        bouncy.state[begin] = y
        bouncy.state[end]   = e
    end
end

# check if bouncy is in the atom
isinatom(b::BouncyMC{NonReversibleBouncy}) = first(b.state) == 0 && last(b.state) < 0
isinatom(b::BouncyMC{ReversibleBouncy}) = first(b.state) == 0

# force a renewal on a Reversible BouncyMC
# send it to the atom and then take a step
function renew!(bouncy::BouncyMC{ReversibleBouncy}, rng::AbstractRNG)
    @inbounds bouncy.state[begin] = zero(eltype(bouncy.state))
    step!(bouncy, rng)
end
# renewal measure is a point mass at (0,+) for nonreversible Bouncy
function renew!(bouncy::BouncyMC{NonReversibleBouncy}, ::AbstractRNG)
    @inbounds begin
        bouncy.state[begin] = zero(eltype(bouncy.state))
        bouncy.state[end]   = one(eltype(bouncy.state))
    end
end

# simulate a tour: starting at (0,+) until absorption into (0,-)
function tour!(bouncy::BouncyMC, rng::AbstractRNG; verbose::Bool=false)
    N = get_N(bouncy)
    renew!(bouncy, rng)
    nhits = 0
    t = 1
    while !isinatom(bouncy)
        t += 1
        verbose && println((t,bouncy.state...))
        step!(bouncy,rng)
        first(bouncy.state) == N && (nhits += 1)
    end
    t, nhits
end

# run iid tours, record tour length and number of hits at the top
function run_tours!(bouncy::BouncyMC, rng::AbstractRNG, times::Vector, counts::Vector; kwargs...)
    @inbounds for k in eachindex(times)
        t, nhits  = tour!(bouncy, rng; kwargs...)
        times[k]  = t      # tour length
        counts[k] = nhits  # number of bounces at the top within the tour
    end
end
function run_tours!(bouncy::BouncyMC{TD,TI}, rng::AbstractRNG; ntours::Int, kwargs...) where {TD,TI}
    times  = Vector{TI}(undef, ntours)
    counts = Vector{TI}(undef, ntours)
    run_tours!(bouncy, rng, times, counts; kwargs...)
    return (times=times, counts=counts)
end
