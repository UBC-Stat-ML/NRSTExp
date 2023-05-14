module CompetingSamplers

using CubicSplines: CubicSpline
using Interpolations
using IrrationalConstants: logtwo
using LogExpFunctions: logsumexp, log1mexp, logaddexp
using OnlineStats: Mean, fit!, value, nobs
using Random: AbstractRNG, TaskLocalRNG, randexp
using SearchSortedNearest: searchsortedprevious, searchsortednext
using Statistics: mean
using StaticArrays: MVector
using UnPack: @unpack
using NRST

include("utils.jl")

include("GeyerThompson1995.jl")
export GT95Sampler

include("SakaiHukushima2016.jl")
export SH16Sampler

include("FaiziEtAl2020.jl")
export FBDRSampler

# quick and dirty way of creating other ST samplers
function NRST.init_sampler(::Type{TST}, args...; kwargs...) where {TST <: NRST.AbstractSTSampler}
    TST(first(NRSTSampler(args...; tune=false, kwargs...)))
end

end