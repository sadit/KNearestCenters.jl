# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module KNearestCenters

using Parameters, LinearAlgebra, CategoricalArrays, StatsBase
using SearchModels, KCenters, SimilaritySearch
import SearchModels: combine, mutate
import StatsBase: predict

export Knc, KncConfig, KncConfigSpace, KncProto, KncProtoConfig, KncProtoConfigSpace, KncPerClassConfigSpace, KncGlobalConfigSpace
export transform, predict

include("scores.jl")  # TODO change supervised learning scores to this package
include("criterions.jl")
include("kernels.jl")

"""
    softmax!(vec::AbstractVector)

Inline computation of the softmax function on the input vector
"""
function softmax!(vec::AbstractVector)
    den = 0.0
    @inbounds @simd for v in vec
        den += exp(v)
    end

    den = 1.0 / den
    @inbounds @simd for i in eachindex(vec)
        vec[i] = exp(vec[i]) * den
    end

    vec
end

include("knc.jl")
include("kncproto.jl")

"""
    transform(nc::Knc, kernel::Function, X, normalize!::Function=softmax!)

Maps a collection of objects to the vector space defined by each center in `nc`; the `kernel` function is used measure the similarity between each ``u \\in X`` and each center in nc. The normalization function is applied to each vector (normalization methods needing to know the attribute's distribution can be applied on the output of `transform`)

"""
function transform(nc::Knc, kernel::Function, X, normalize!::Function=softmax!)
    KCenters.transform(nc.centers, nc.dmax, kernel, X, normalize!)
end

end