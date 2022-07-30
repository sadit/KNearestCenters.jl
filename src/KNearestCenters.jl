# This file is a part of KNearestCenters.jl

module KNearestCenters

using StatsBase: mean, mode
using LinearAlgebra, CategoricalArrays
using SearchModels, KCenters, SimilaritySearch
import SearchModels: combine, mutate
import StatsAPI: predict, fit
using MLUtils

export Knc, KncConfig, KncConfigSpace, KncProto, KncProtoConfig, KncProtoConfigSpace, KncPerClassConfigSpace, KncGlobalConfigSpace
export transform, predict, fit, categorical

include("scores.jl") 
include("criterions.jl")
include("kernels.jl")
include("knn.jl")
include("knnopt.jl")
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