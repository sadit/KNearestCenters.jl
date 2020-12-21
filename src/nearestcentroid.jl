# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SimilaritySearch
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict
using CategoricalArrays
export KNC, fit, predict, transform, most_frequent_label, mean_label

"""
A simple nearest centroid classifier with support for kernel functions
"""
mutable struct KNC{T}
    centers::Vector{T}
    dmax::Vector{Float64}
    class_map::Vector{Int}
    nclasses::Int
end

"""
    fit(::Type{KNC}, D::DeloneHistogram, class_map::Vector{Int}=Int[]; verbose=true)
    fit(::Type{KNC}, D::DeloneInvIndex, labels::AbstractVector; verbose=true)
    fit(::Type{KNC}, dist::Function, input_clusters::NamedTuple, train_X::AbstractVector, train_y::AbstractVector{_Integer}, centroid::Function=mean; split_entropy=0.1, verbose=false) where _Integer<:Integer

Creates a KNC classifier using the output of either `kcenters` or `kcenters` as input
through either a `DeloneHistogram` or `DeloneInvIndex` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.

The third form is a little bit more complex, the idea is to divide clusters whenever their label-diversity surpasses a given threshold (measured with `split_entropy`).
This function receives a distance function `dist` and the original dataset `train_X` in addition to other mentioned arguments.
"""
function fit(::Type{KNC}, D::DeloneHistogram, class_map::Vector{Int}, verbose=true)
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    KNC(D.centers.db, D.dmax, class_map, length(unique(class_map)))
end

function fit(::Type{KNC}, C::NamedTuple, class_map::Vector{Int}=Int[]; verbose=true)
    D = fit(DeloneHistogram, C)
    fit(KNC, D, class_map)
end

function fit(
    ::Type{KNC},
    dist::Function,
    input_clusters::NamedTuple,
    train_X::AbstractVector,
    train_y::CategoricalArray,
    centroid::Function=mean;
    split_entropy=0.3,
    minimum_elements_per_centroid=1,
    verbose=false
)    
    centroids = eltype(train_X)[] # clusters
    classes = Int[] # class mapping between clusters and classes
    dmax = Float64[]
    ncenters = length(input_clusters.centroids)
    nclasses = length(levels(train_y))
    _ent(f, n) = (f == 0) ? 0.0 : (f / n * log(n / f))
    
    for centerID in 1:ncenters
        lst = Int[pos for (pos, c) in enumerate(input_clusters.codes) if c == centerID] # objects related to this centerID
        ylst = @view train_y.refs[lst] # real labels related to this centerID
        freqs = counts(ylst, 1:nclasses) # histogram of labels in this centerID
        @info freqs
        labels = findall(f -> f >= minimum_elements_per_centroid, freqs)

        # compute entropy of the set of labels
        # skip if the minimum number of elements is not reached
        if length(labels) == 0
            verbose && println(stderr, "*** center $centerID: ignoring all elements because minimum-frequency restrictions were not met, freq >= $minimum_elements_per_centroid, freqs: $freqs")
            continue
        end

        verbose && println(stderr, "*** center $centerID: selecting labels $labels (freq >= $minimum_elements_per_centroid) [from $freqs]")

        if length(labels) == 1
            # a unique label, skip computation
            e = 0.0
        else
            freqs_ = freqs[labels]
            n = sum(freqs_)
            e = sum(_ent(f, n) for f in freqs_) / log(length(labels))
            verbose && println(stderr, "** centroid: $centerID, normalized-entropy: $e, ", 
                collect(zip(labels, freqs_)))
        end

        if e > split_entropy            
            for l in labels
                XX = [train_X[lst[pos]] for (pos, c) in enumerate(ylst) if c == l]
                c = centroid(XX)
                push!(centroids, c)
                push!(classes, l)
                d = 0.0
                for u in XX
                    d = max(d, convert(Float64, dist(u, c)))
                end

                push!(dmax, d)
            end
        else
            push!(centroids, input_clusters.centroids[centerID])
            freq, pos = findmax(freqs)
            push!(classes, pos)
            d = 0.0
            for objID in lst
                d = max(d, convert(Float64, dist(train_X[objID], centroids[end])))
            end
            push!(dmax, d)
         end
    end

    verbose && println(stderr, "finished with $(length(centroids)) centroids; started with $(length(input_clusters.centroids))")
    KNC(centroids, dmax, classes, nclasses)
end

"""
    most_frequent_label(nc::KNC, res::KnnResult)

Summary function that computes the label as the most frequent label among labels of the k nearest prototypes (categorical labels)
"""
function most_frequent_label(nc::KNC, res::KnnResult)
    c = counts([nc.class_map[p.id] for p in res], 1:nc.nclasses)
    findmax(c)[end]
end

"""
    mean_label(nc::KNC, res::KnnResult)

Summary function that computes the label as the mean of the k nearest labels (ordinal classification)
"""
function mean_label(nc::KNC, res::KnnResult)
    round(Int, mean([nc.class_map[p.id] for p in res]))
end

"""
    predict(nc::KNC{PointType}, kernel::Function, summary::Function, X::AbstractVector{PointType}) where PointType
    predict(nc::KNC{PointType}, kernel::Function, summary::Function, x::PointType, k::Integer) where PointType

Predicts the class of `x` using the label of the `k` nearest centroid under the `kernel` function.
"""
function predict(nc::KNC{PointType}, kernel::Function, summary::Function, X::AbstractVector{PointType}, k::Integer) where PointType
    res = KnnResult(k)
    ypred = Vector{Int}(undef, length(X))

    for j in eachindex(X)
        empty!(res)
        ypred[j] = predict(nc, kernel, summary, X[j], k, res)
    end

    ypred
end

function predict(nc::KNC{PointType}, kernel::Function, summary::Function, x::PointType, k::Integer, res::KnnResult) where PointType
    C = nc.centers
    dmax = nc.dmax

    for i in eachindex(C)
        s = eval_kernel(kernel, x, C[i], dmax[i])
        push!(res, i, -s)
    end

    summary(nc, res)
end

function predict(nc::KNC{PointType}, kernel::Function, summary::Function, x::PointType, k::Integer) where PointType
    predict(nc, kernel, summary, x, k, KnnResult(k))
end

"""
    eval_kernel(kernel::Function, a, b, σ)

Evaluates a kernel function over the giver arguments (isolated to ensure that the function can be compiled)
"""
function eval_kernel(kernel::Function, a, b, σ)
    kernel(a, b, σ)
end

function broadcastable(nc::KNC)
    (nc,)
end

