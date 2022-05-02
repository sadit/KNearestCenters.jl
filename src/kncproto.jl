# This file is a part of KNearestCenters.jl

@with_kw struct KncProtoConfig{K_<:AbstractKernel, S_<:AbstractCenterSelection}
    kernel::K_ = ReluKernel(L2Distance())
    centerselection::S_ = CentroidSelection()

    k::Int32 = 1
    ncenters::Int32 = 7
    maxiters::Int32 = 1

    recall::Float32 = 1.0
    initial_clusters = :rand
    split_entropy::Float32 = 0.5
    minimum_elements_per_region::Int = 3
end

config_type(knc::KncProtoConfig) = (KncProtoConfig, sign(knc.ncenters))

@with_kw struct KncProtoConfigSpace{KindProto,MutProb} <: AbstractSolutionSpace
    kernel = [k_(d_()) for k_ in [DirectKernel, ReluKernel, GaussianKernel], d_ in [L2Distance, CosineDistance]]
    centerselection = [CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()]
    k = 1:1:7
    maxiters = 1:3:15
    recall = [1.0]
    ncenters = 2:33
    initial_clusters = [:fft, :dnet, :rand]
    split_entropy = 0.1:0.1:0.9
    minimum_elements_per_region = 1:3:30
    scale_k = (lower=1, upper=11, s=1.5, p1=MutProb)
    scale_maxiters = (lower=1, upper=30, s=1.5, p1=MutProb)
    scale_recall = (lower=0.1, upper=1.0, s=1.5, p1=MutProb)
    scale_split_entropy = (lower=0.1, upper=1.0, s=1.5, p1=MutProb)
    scale_minimum_elements_per_region = (lower=1, upper=30, s=1.5, p1=MutProb)
end

const KncPerClassConfigSpace{MutProb} = KncProtoConfigSpace{-1,MutProb}
const KncGlobalConfigSpace{MutProb} = KncProtoConfigSpace{1,MutProb}

Base.eltype(::KncProtoConfigSpace) = KncProtoConfig

_fix_ncenters(ncenters) = ncenters in (0, 1) ? 2 : ncenters

"""
    rand(space::KncProtoConfigSpace)

Creates a random `KncProtoConfig` instance based on the `space` definition.
"""
function Base.rand(space::KncProtoConfigSpace{KindProto,MutProb}) where {KindProto,MutProb}
    ncenters = _fix_ncenters(rand(space.ncenters) * KindProto)
    config = KncProtoConfig(
        rand(space.kernel),
        rand(space.centerselection),
        rand(space.k),
        ncenters,
        rand(space.maxiters),
        rand(space.recall),
        rand(space.initial_clusters),
        ncenters < 0 ? 1.0 : rand(space.split_entropy),
        rand(space.minimum_elements_per_region)
    )
end

"""
    combine(a::KncProtoConfig, b::KncProtoConfig)

Creates a new configuration combining the given configurations
"""
function combine(a::KncProtoConfig, b::KncProtoConfig)
    KncProtoConfig(
        b.kernel,
        b.centerselection,
        a.k,
        a.ncenters,
        a.maxiters,
        b.recall,
        a.initial_clusters,
        a.split_entropy,
        a.minimum_elements_per_region,
    )
end

"""
    mutate(space::KncProtoConfigSpace, a::KncProtoConfig, iter)

Creates a new configuration based on a slight perturbation of `a`
"""
function mutate(space::KncProtoConfigSpace, a::KncProtoConfig, iter)
    KncProtoConfig(
        SearchModels.change(a.kernel, space.kernel),
        SearchModels.change(a.centerselection, space.centerselection),
        SearchModels.scale(a.k; space.scale_k...),
        a.ncenters,
        SearchModels.scale(a.k; space.scale_maxiters...),
        SearchModels.scale(a.k; space.scale_recall...),
        SearchModels.change(a.initial_clusters, space.initial_clusters),
        SearchModels.scale(a.k; space.scale_split_entropy...),
        SearchModels.scale(a.k; space.scale_minimum_elements_per_region...)
    )
end


#################################
### Definition of the classifier
#################################

"""
A simple nearest centroid classifier with support for kernel functions
"""
struct KncProto{DataType<:AbstractDatabase, K_<:KncProtoConfig}
    config::K_
    centers::DataType
    dmax::Vector{Float32}
    class_map::Vector{Int32}
    nclasses::Int32
    res::KnnResult
end

"""
    KncProto(config::KncProtoConfig, X, y::CategoricalArray; verbose=true)
    KncProto(config::KncProtoConfig,
        input_clusters::ClusteringData,
        train_X::AbstractVector,
        train_y::CategoricalArray;
        verbose=false
    )

Creates a KncProto classifier using the given configuration and data.
"""
function KncProto(config::KncProtoConfig, X::AbstractDatabase, y::CategoricalArray; verbose=true)
    config.ncenters == 0 && error("invalid ncenter $ncenters; ncenters <= -2 or 2 <= ncenters; please use plain Knc otherwise")
    if config.ncenters > 0
        # computes a set of ncenters for all dataset
        verbose && println(stderr, "KncProto> clustering data without knowing labels", config)
        D = kcenters(config.kernel.dist, X, config.ncenters;
                sel=config.centerselection,
                initial=config.initial_clusters,
                recall=config.recall,
                verbose=verbose,
                maxiters=config.maxiters)
        KncProto(config, D, X, y; verbose=verbose)
    else
        # computes a set of ncenters centers for each label
        ncenters = abs(config.ncenters)
        verbose && println(stderr, "KncProto> clustering data with label division", config)
        nclasses = length(levels(y))
        centers = eltype(X)[]
        dmax = Float32[]
        class_map = Int32[]
        nclasses = length(levels(y))

        M = labelmap(y.refs)
        for ilabel in 1:nclasses
            L = get(M, ilabel, nothing)
            L === nothing && continue
            X_ = SubDatabase(X, L)
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

        KncProto(config, VectorDatabase(centers), dmax, class_map, convert(Int32, nclasses), KnnResult(1))
    end
end

function KncProto(
        config::KncProtoConfig,
        input_clusters::ClusteringData,
        train_X::AbstractDatabase,
        train_y::CategoricalArray;
        verbose=false
    )
    train_X = convert(AbstractDatabase, train_X)
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
                XX = SubDatabase(train_X, [lst[pos] for (pos, c) in enumerate(ylst) if c == l])
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
            c = input_clusters.centers[centerID]
            push!(centers, c)
            freq, pos = findmax(freqs)
            push!(classes, pos)
            d = 0.0
            for objID in lst
                d_ = evaluate(config.kernel.dist, train_X[objID], c)
                d = max(d, convert(Float64, d_))
            end
            push!(dmax, d)
         end
    end

    verbose && println(stderr, "finished with $(length(centers)) centers; started with $(length(input_clusters.centers))")
    KncProto(config, VectorDatabase(centers), dmax, classes, convert(Int32, nclasses), KnnResult(1))
end

"""
    most_frequent_label(nc::KncProto, res::KnnResult)

Summary function that computes the label as the most frequent label among labels of the k nearest prototypes (categorical labels)
"""
function most_frequent_label(nc::KncProto, res::KnnResult)
    c = counts([nc.class_map[id] for id in idview(res)], 1:nc.nclasses)
    findmax(c)[end]
end

"""
    mean_label(nc::KncProto, res::KnnResult)

Summary function that computes the label as the mean of the k nearest labels (ordinal classification)
"""
function mean_label(nc::KncProto, res::KnnResult)
    round(Int, mean([nc.class_map[id] for (id, dist) in res]))
end

"""
    predict(nc::KncProto, x, res::KnnResult)
    predict(nc::KncProto, x)

Predicts the class of `x` using the label of the `k` nearest centers under the `kernel` function.
"""
function predict(nc::KncProto, x, res::KnnResult=reuse!(nc.res))
    C = nc.centers
    dmax = nc.dmax
    for i in eachindex(C)
        s = evaluate(nc.config.kernel, x, C[i], dmax[i])
        push!(res, i, -s)
    end

    most_frequent_label(nc, res)
end

function Base.broadcastable(nc::KncProto)
    (nc,)
end

