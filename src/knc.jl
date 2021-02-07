# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using LinearAlgebra
using StatsBase
import StatsBase: fit, predict
using CategoricalArrays
export Knc, fit, predict, transform

"""
A simple nearest centroid classifier with support for kernel functions
"""
struct Knc{DataType<:AbstractVector, K_<:KncConfig}
    config::K_
    centers::DataType
    dmax::Vector{Float32}
    class_map::Vector{Int32}
    nclasses::Int32
    res::KnnResult
end

StructTypes.StructType(::Type{<:Knc}) = StructTypes.Struct()

"""
    Knc(config::KncConfig, X, y::CategoricalArray; verbose=true)
    Knc(config::KncConfig,
        input_clusters::ClusteringData,
        train_X::AbstractVector,
        train_y::CategoricalArray;
        verbose=false
    )

Creates a Knc classifier using the given configuration and data.
"""
function Knc(config::KncConfig, X, y::CategoricalArray; verbose=true)
    if config.ncenters == 0
        # computes a set of #labels centers using labels for clustering
        verbose && println("Knc> clustering data with labels")
        D = kcenters(config.kernel.dist, X, y, config.centerselection)
        nclasses = length(levels(y))
        @assert nclasses <= length(D.centers)
        class_map = collect(Int32, 1:nclasses)
        Knc(config, D.centers, D.dmax, class_map, convert(Int32, nclasses), KnnResult(1))
    elseif config.ncenters > 0
        # computes a set of ncenters for all dataset
        verbose && println("Knc> clustering data without knowing labels")
        D = kcenters(config.kernel.dist, X, config.ncenters;
                sel=config.centerselection,
                initial=config.initial_clusters,
                recall=config.recall,
                verbose=verbose,
                maxiters=config.maxiters)
        Knc(config, D, X, y; verbose=verbose)
    else
        # computes a set of ncenters centers for each label
        ncenters = abs(config.ncenters)
        verbose && println("Knc> clustering data with label division")
        nclasses = length(levels(y))
        centers = eltype(X)[]
        dmax = Float32[]
        class_map = Int32[]
        nclasses = length(levels(y))

        for ilabel in 1:nclasses
            mask = y.refs .== ilabel
            X_ = X[mask]
            D = kcenters(config.kernel.dist, X_, ncenters;
                    sel=config.centerselection,
                    initial=config.initial_clusters,
                    recall=config.recall,
                    verbose=verbose,
                    maxiters=config.maxiters)
            
            for i in eachindex(D.centers)
                if D.freqs[i] >= config.minimum_elements_per_region
                    push!(centers, D.centers[i])
                    push!(dmax, D.dmax[i])
                    push!(class_map, ilabel)
                end
            end
        end

        Knc(config, centers, dmax, class_map, convert(Int32, nclasses), KnnResult(1))
    end
end

function Knc(
        config::KncConfig,
        input_clusters::ClusteringData,
        train_X::AbstractVector,
        train_y::CategoricalArray;
        verbose=false
    )
    centers = eltype(train_X)[] # clusters
    classes = Int32[] # class mapping between clusters and classes
    dmax = Float32[]
    ncenters = length(input_clusters.centers)
    nclasses = length(levels(train_y))
    _ent(f, n) = (f == 0) ? 0.0 : (f / n * log(n / f))
    
    for centerID in 1:ncenters
        lst = Int[pos for (pos, c) in enumerate(input_clusters.codes) if c == centerID] # objects related to this centerID
        ylst = @view train_y.refs[lst] # real labels related to this centerID
        freqs = counts(ylst, 1:nclasses) # histogram of labels in this centerID
        @info freqs
        labels = findall(f -> f >= config.minimum_elements_per_region, freqs)

        # compute entropy of the set of labels
        # skip if the minimum number of elements is not reached
        if length(labels) == 0
            verbose && println(stderr, "*** center $centerID: ignoring all elements because minimum-frequency restrictions were not met, freq >= $(config.minimum_elements_per_region), freqs: $freqs")
            continue
        end

        verbose && println(stderr, "*** center $centerID: selecting labels $labels (freq >= $(config.minimum_elements_per_region)) [from $freqs]")

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

        if e > config.split_entropy            
            for l in labels
                XX = [train_X[lst[pos]] for (pos, c) in enumerate(ylst) if c == l]
                c = center(config.centerselection, XX)
                push!(centers, c)
                push!(classes, l)
                d = 0.0
                for u in XX
                    d = max(d, convert(Float64, evaluate(config.kernel.dist, u, c)))
                end

                push!(dmax, d)
            end
        else
            push!(centers, input_clusters.centers[centerID])
            freq, pos = findmax(freqs)
            push!(classes, pos)
            d = 0.0
            for objID in lst
                d_ = evaluate(config.kernel.dist, train_X[objID], centers[end])
                d = max(d, convert(Float64, d_))
            end
            push!(dmax, d)
         end
    end

    verbose && println(stderr, "finished with $(length(centers)) centers; started with $(length(input_clusters.centers))")
    Knc(config, centers, dmax, classes, convert(Int32, nclasses), KnnResult(1))
end

"""
    most_frequent_label(nc::Knc, res::KnnResult)

Summary function that computes the label as the most frequent label among labels of the k nearest prototypes (categorical labels)
"""
function most_frequent_label(nc::Knc, res::KnnResult)
    c = counts([nc.class_map[p.id] for p in res], 1:nc.nclasses)
    findmax(c)[end]
end

"""
    mean_label(nc::Knc, res::KnnResult)

Summary function that computes the label as the mean of the k nearest labels (ordinal classification)
"""
function mean_label(nc::Knc, res::KnnResult)
    round(Int, mean([nc.class_map[p.id] for p in res]))
end

"""
    predict(nc::Knc, x, res::KnnResult)
    predict(nc::Knc, x)

Predicts the class of `x` using the label of the `k` nearest centers under the `kernel` function.
"""
function predict(nc::Knc, x, res::KnnResult)
    C = nc.centers
    dmax = nc.dmax
    for i in eachindex(C)
        s = evaluate(nc.config.kernel, x, C[i], dmax[i])
        push!(res, i, -s)
    end

    most_frequent_label(nc, res)
end

function predict(nc::Knc, x)
    empty!(nc.res)
    predict(nc, x, nc.res)
end

function Base.broadcastable(nc::Knc)
    (nc,)
end

