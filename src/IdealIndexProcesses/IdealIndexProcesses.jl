module IdealIndexProcesses

using Distributions: Exponential
using StaticArrays: MVector
using FillArrays: Fill
using Random: AbstractRNG
import NRST: renew!, tour!, step!, isinatom, toatom!, get_N

abstract type Bouncy end

export BouncyPDMP
include("BouncyPDMP.jl") # PDMP with reflective boundaries in [0,1]

export BouncyMC, NonReversibleBouncy, ReversibleBouncy
include("BouncyMC.jl")   # Markov chain on 0:N × {-1,1}

# common
export run_tours!

end # end module IdealIndexProcesses
