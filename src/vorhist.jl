# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
export DeloneHistogram, fit, predict

mutable struct DeloneHistogram{T}
    _type::Type{T}
    centers::Index
    freqs::Vector{Int}
    dmax::Vector{Float64}
    n::Int
end

function fit(::Type{DeloneHistogram}, kcenters_::NamedTuple)
    k = length(kcenters_.centroids)
    freqs = zeros(Int, k)
    dmax = zeros(Float64, k)
    
    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        freqs[code] += 1
        dmax[code] = max(dmax[code], d)
    end

    C = fit(Sequential, kcenters_.centroids)
    T = eltype(kcenters_.centroids)
    DeloneHistogram{T}(T, C, freqs, dmax, length(kcenters_.codes))
end

function predict(vor::DeloneHistogram{T}, dist::Function, q::T) where T
    predict(vor, dist, [q])[1]
end

function predict(vor::DeloneHistogram{T}, dist::Function, queries::AbstractVector{T}) where T
    res = KnnResult(1)
    L = Vector{Int}(undef, length(queries))

    for i in eachindex(queries)
        empty!(res)
        search(vor.centers, dist, queries[i], res)
        c = first(res).id
        sim = max(0.0, 1.0 - first(res).dist  / vor.dmax[c])
        L[i] = sim > 0
    end

    L
end
