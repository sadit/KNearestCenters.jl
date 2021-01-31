# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SimilaritySearch
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict
using CategoricalArrays
export KNC, fit, predict, transform

"""
A simple nearest centroid classifier with support for kernel functions
"""
struct KNC{DataType<:AbstractVector, KernelType<:AbstractKernel}
    kernel::KernelType
    centers::DataType
    dmax::Vector{Float64}
    class_map::Vector{Int}
    nclasses::Int
    res::KnnResult
end

StructTypes.StructType(::Type{<:KNC}) = StructTypes.Struct()

"""
    KNC(kernel::AbstractKernel, D::DeloneHistogram, class_map::Vector{Int})
    KNC(kernel::AbstractKernel, C::ClusteringData, class_map::Vector{Int}=Int[])
    KNC(
        dist::AbstractKernel,
        input_clusters::ClusteringData,
        train_X::AbstractVector,
        train_y::CategoricalArray;
        centerselection::AbstractCenterSelection=CentroidSelection(),
        split_entropy=0.3,
        minimum_elements_per_region=1,
        verbose=false
    ) 

Creates a KNC classifier using the output of either `kcenters` or `kcenters` as input
through a `DeloneHistogram` struct.
If `class_map` is given, then it contains the list of labels to be reported associated to centers; if they are not specified,
then they are assigned in consecutive order for `DeloneHistogram` and as the most popular label for `DeloneInvIndex`.

The third form is a little bit more complex, the idea is to divide clusters whenever their label-diversity surpasses a given threshold (measured with `split_entropy`).
This function receives a distance function `dist` and the original dataset `train_X` in addition to other mentioned arguments.
"""
function KNC(kernel::AbstractKernel, D::DeloneHistogram, class_map::Vector{Int})
    if length(class_map) == 0
        class_map = collect(1:length(D.centers.db))
    end

    KNC(kernel, D.centers.db, D.dmax, class_map, length(unique(class_map)), KnnResult(1))
end

function KNC(kernel::AbstractKernel, C::ClusteringData, class_map::Vector{Int}=Int[])
    D = DeloneHistogram(kernel.dist, C)
    KNC(kernel, D, class_map)
end

function KNC(
    kernel::AbstractKernel,
    input_clusters::ClusteringData,
    train_X::AbstractVector,
    train_y::CategoricalArray;
    centerselection::AbstractCenterSelection=CentroidSelection(),
    split_entropy=0.3,
    minimum_elements_per_region=1,
    verbose=false
)    
    centers = eltype(train_X)[] # clusters
    classes = Int[] # class mapping between clusters and classes
    dmax = Float64[]
    ncenters = length(input_clusters.centers)
    nclasses = length(levels(train_y))
    _ent(f, n) = (f == 0) ? 0.0 : (f / n * log(n / f))
    
    for centerID in 1:ncenters
        lst = Int[pos for (pos, c) in enumerate(input_clusters.codes) if c == centerID] # objects related to this centerID
        ylst = @view train_y.refs[lst] # real labels related to this centerID
        freqs = counts(ylst, 1:nclasses) # histogram of labels in this centerID
        @info freqs
        labels = findall(f -> f >= minimum_elements_per_region, freqs)

        # compute entropy of the set of labels
        # skip if the minimum number of elements is not reached
        if length(labels) == 0
            verbose && println(stderr, "*** center $centerID: ignoring all elements because minimum-frequency restrictions were not met, freq >= $minimum_elements_per_region, freqs: $freqs")
            continue
        end

        verbose && println(stderr, "*** center $centerID: selecting labels $labels (freq >= $minimum_elements_per_region) [from $freqs]")

        if length(labels) == 1
            # a unique label, skip computation
            e = 0.0
        else
            freqs_ = freqs[labels]
            n = sum(freqs_)
            e = sum(_ent(f, n) for f in freqs_) / log(length(labels))
            verbose && println(stderr, "** center: $centerID, normalized-entropy: $e, ", 
                collect(zip(labels, freqs_)))
        end

        if e > split_entropy            
            for l in labels
                XX = [train_X[lst[pos]] for (pos, c) in enumerate(ylst) if c == l]
                c = center(centerselection, XX)
                push!(centers, c)
                push!(classes, l)
                d = 0.0
                for u in XX
                    d = max(d, convert(Float64, evaluate(kernel.dist, u, c)))
                end

                push!(dmax, d)
            end
        else
            push!(centers, input_clusters.centers[centerID])
            freq, pos = findmax(freqs)
            push!(classes, pos)
            d = 0.0
            for objID in lst
                d_ = evaluate(kernel.dist, train_X[objID], centers[end])
                d = max(d, convert(Float64, d_))
            end
            push!(dmax, d)
         end
    end

    verbose && println(stderr, "finished with $(length(centers)) centers; started with $(length(input_clusters.centers))")
    KNC(kernel, centers, dmax, classes, nclasses, KnnResult(1))
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
    predict(nc::KNC, x, res::KnnResult)
    predict(nc::KNC, x)

Predicts the class of `x` using the label of the `k` nearest centers under the `kernel` function.
"""
function predict(nc::KNC, x, res::KnnResult)
    C = nc.centers
    dmax = nc.dmax
    for i in eachindex(C)
        s = evaluate(nc.kernel, x, C[i], dmax[i])
        push!(res, i, -s)
    end

    most_frequent_label(nc, res)
end

function predict(nc::KNC, x)
    empty!(nc.res)
    predict(nc, x, nc.res)
end

function broadcastable(nc::KNC)
    (nc,)
end

