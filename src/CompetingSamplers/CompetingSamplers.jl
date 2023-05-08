module CompetingSamplers

using IrrationalConstants: logtwo
using LogExpFunctions: logsumexp, log1mexp, logaddexp
using OnlineStats: Mean, fit!, value, nobs
using Random: AbstractRNG, TaskLocalRNG, randexp
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

end