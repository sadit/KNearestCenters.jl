# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
export DeloneHistogram, fit, predict

mutable struct DeloneHistogram{CentersSearchType<:AbstractSearchContext}
    centers::CentersSearchType
    freqs::Vector{Int}
    dmax::Vector{Float64}
    n::Int
end

StructTypes.StructType(::Type{<:DeloneHistogram}) = StructTypes.Struct()

function DeloneHistogram(dist::PreMetric, kcenters_::ClusteringData)
    k = length(kcenters_.centers)
    freqs = zeros(Int, k)
    dmax = zeros(Float64, k)
    
    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        freqs[code] += 1
        dmax[code] = max(dmax[code], d)
    end

    C = ExhaustiveSearch(dist, kcenters_.centers)
    DeloneHistogram(C, freqs, dmax, length(kcenters_.codes))
end

function predict(vor::DeloneHistogram, q)
    res = search(vor.centers, q)
    c = first(res).id
    sim = max(0.0, 1.0 - first(res).dist  / vor.dmax[c])
    sim > 0
end
