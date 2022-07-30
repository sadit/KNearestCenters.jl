# This file is a part of KNearestCenters.jl

Base.@kwdef struct KncProtoConfig{D_<:SemiMetric, S_<:AbstractCenterSelection}
    dist::D_ = L2Distance()
    centerselection::S_ = CentroidSelection()

    k::Int = 1
    ncenters::Int = 7
    maxiters::Int = 1

    initial_clusters = :rand
    split_entropy::Float64 = 0.5
    minimum_elements_per_region::Int = 3
end

config_type(knc::KncProtoConfig) = (KncProtoConfig, sign(knc.ncenters))

Base.@kwdef struct KncProtoConfigSpace{MutProb} <: AbstractSolutionSpace
    dist = [L1Distance(), L2Distance(), CosineDistance()]
    centerselection = [CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()]
    k = 1:1:7
    maxiters = 1:3:15
    ncenters = 2:33
    initial_clusters = [:fft, :dnet, :rand]
    split_entropy = 0.1:0.1:0.9
    minimum_elements_per_region = 1:3:30
    scale_k = (lower=1, upper=11, s=1.5, p1=MutProb)
    scale_maxiters = (lower=1, upper=30, s=1.5, p1=MutProb)
    scale_split_entropy = (lower=0.1, upper=1.0, s=1.5, p1=MutProb)
    scale_minimum_elements_per_region = (lower=1, upper=30, s=1.5, p1=MutProb)
end

Base.eltype(::KncProtoConfigSpace) = KncProtoConfig

_fix_ncenters(ncenters) = ncenters in (0, 1) ? 2 : ncenters

"""
    rand(space::KncProtoConfigSpace)

Creates a random `KncProtoConfig` instance based on the `space` definition.
"""
function Base.rand(space::KncProtoConfigSpace{MutProb}) where {MutProb}
    ncenters = _fix_ncenters(rand(space.ncenters))
    KncProtoConfig(
        rand(space.dist),
        rand(space.centerselection),
        rand(space.k),
        ncenters,
        rand(space.maxiters),
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
        rand((a.dist, b.dist)),
        rand((a.centerselection, b.centerselection)),
        rand((a.k, b.k)),
        rand((a.ncenters, b.ncenters)),
        rand((a.maxiters, b.maxiters)),
        rand((a.initial_clusters, b.initial_clusters)),
        rand((a.split_entropy, b.split_entropy)),
        rand((a.minimum_elements_per_region, b.minimum_elements_per_region))
    )
end

"""
    mutate(space::KncProtoConfigSpace, a::KncProtoConfig, iter)

Creates a new configuration based on a slight perturbation of `a`
"""
function mutate(space::KncProtoConfigSpace, a::KncProtoConfig, iter)
    KncProtoConfig(
        SearchModels.change(a.dist, space.dist),
        SearchModels.change(a.centerselection, space.centerselection),
        SearchModels.scale(a.k; space.scale_k...),
        a.ncenters,
        SearchModels.scale(a.maxiters; space.scale_maxiters...),
        SearchModels.change(a.initial_clusters, space.initial_clusters),
        SearchModels.scale(a.split_entropy; space.scale_split_entropy...),
        SearchModels.scale(a.minimum_elements_per_region; space.scale_minimum_elements_per_region...)
    )
end

#################################
### Definition of the classifier
#################################

"""
A simple nearest centroid classifier
"""
struct KncProto{K_<:KncProtoConfig,KM_<:KnnModel}
    config::K_
    nclasses::Int
    knn::KM_
end

"""
    fit(config::KncProtoConfig, X, y::CategoricalArray; verbose=true)
    fit(config::KncProtoConfig,
        input_clusters::ClusteringData,
        train_X::AbstractVector,
        train_y::CategoricalArray;
        verbose=false
    )

Creates a KncProto classifier using the given configuration and data.
"""
function fit(config::KncProtoConfig, X::AbstractDatabase, y::CategoricalArray; verbose=true, loss=BalancedErrorRate())
    ncenters = config.ncenters
    ncenters > 1 || error("invalid ncenter $ncenters; 2 < ncenters; please use plain Knc otherwise")

    # computes a set of ncenters for all dataset
    verbose && println(stderr, "KncProto> clustering data without knowing labels", config)
    D = kcenters(config.dist, X, config.ncenters;
            sel=config.centerselection,
            initial=config.initial_clusters,
            recall=1.0,
            verbose=verbose,
            maxiters=config.maxiters)
    
    fit(config, D, X, y; verbose=verbose)
end

function fit(
        config::KncProtoConfig,
        input_clusters::ClusteringData,
        train_X::AbstractDatabase,
        train_y::CategoricalArray;
        loss=BalancedErrorRate(),
        verbose=false
    )
    train_X = convert(AbstractDatabase, train_X)
    centers = eltype(train_X)[] # clusters
    Levels_ = levels(train_y)
    class_map = eltype(Levels_)[] # class mapping between clusters and classes
    ncenters = length(input_clusters.centers)
    nclasses = length(Levels_)
    n = length(train_y)
    0 < n && nclasses < n || throw(InvalidSetupError(config, "must follow 0 < $n && $nclasses < $n"))
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
                push!(class_map, Levels_[l])
            end
        else
            c = input_clusters.centers[centerID]
            push!(centers, c)
            freq, pos = findmax(freqs)
            push!(class_map, Levels_[pos])
         end
    end

    verbose && println(stderr, "finished with $(length(centers)) centers; started with $(length(input_clusters.centers))")
    knn = fit(KnnModel, VectorDatabase(centers), categorical(class_map))
    optimize!(knn, loss)
    KncProto(config, nclasses, knn)
end

"""
    predict(nc::KncProto, x)

Predicts the class of `x` using the label of the `k` nearest centers
"""
predict(nc::KncProto, x) = predict(nc.knn, x)

function Base.broadcastable(nc::KncProto)
    (nc,)
end
