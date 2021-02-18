# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
# based on Rocchio implementation (rocchio.jl) of https://github.com/sadit/TextSearch.jl

using SearchModels
import SearchModels: random_configuration, combine_configurations
export KncConfigSpace, KncConfig, KCenters

struct KncConfig{K_<:AbstractKernel, S_<:AbstractCenterSelection}
    kernel::K_
    centerselection::S_

    k::Int
    ncenters::Int
    maxiters::Int

    recall::Float64
    initial_clusters
    split_entropy::Float64
    minimum_elements_per_region::Int
end

StructTypes.StructType(::Type{<:KncConfig}) = StructTypes.Struct()

KncConfig(;
    kernel=ReluKernel(CosineDistance()),
    centerselection::AbstractCenterSelection=CentroidSelection(),

    k::Int=1,
    ncenters::Integer=0,
    maxiters::Integer=1,
    
    recall::AbstractFloat=1.0,
    initial_clusters=:rand,
    split_entropy::AbstractFloat=0.6,
    minimum_elements_per_region=3
) = KncConfig(
        kernel, centerselection, k, ncenters, maxiters,
        recall, initial_clusters, split_entropy, minimum_elements_per_region)

struct KncConfigSpace <: AbstractSolutionSpace
    kernel::Array{AbstractKernel}
    centerselection::Vector{AbstractCenterSelection}
    k::Vector{Integer}
    maxiters::Vector{Integer}
    recall::Vector{Real}
    ncenters::Vector{Integer}
    initial_clusters::Vector{Any}
    split_entropy::Vector{Real}
    minimum_elements_per_region::Vector{Integer}
end

Base.eltype(::KncConfigSpace) = KncConfig

"""
    KncConfigSpace(;
        kernel::Array=[k_(d_()) for k_ in [DirectKernel, ReluKernel],
                                    d_ in [L2Distance, CosineDistance]],                         
        centerselection::AbstractVector=[CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()],
        k::AbstractVector=[1],
        maxiters::AbstractVector=[1, 3, 10],
        recall::AbstractVector=[1.0],
        ncenters::AbstractVector=[0, 10],
        initial_clusters::AbstractVector=[:fft, :dnet, :rand],
        split_entropy::AbstractVector=[0.3, 0.6, 0.9],
        minimum_elements_per_region::AbstractVector=[1, 3, 5]
    )

Creates a configuration space for KncConfig
"""
KncConfigSpace(;
    kernel::Array{AbstractKernel}=[k_(d_()) for k_ in [DirectKernel, ReluKernel],
                                                d_ in [L2Distance, CosineDistance]],
    centerselection::AbstractVector=[CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()],
    k::Vector=[1],
    maxiters::Vector=[1, 3, 10],
    recall::Vector=[1.0],
    ncenters::Vector=[0, 10],
    initial_clusters::Vector=[:fft, :dnet, :rand],
    split_entropy::Vector=[0.3, 0.6, 0.9],
    minimum_elements_per_region::Vector=[1, 3, 5]
) = KncConfigSpace(kernel, centerselection, k, maxiters, recall, ncenters, initial_clusters, split_entropy, minimum_elements_per_region)

"""
    random_configuration(space::KncConfigSpace)

Creates a random `KncConfig` instance based on the `space` definition.
"""
function random_configuration(space::KncConfigSpace)
    ncenters = rand(space.ncenters)

    if ncenters == 0
        maxiters = 0
        split_entropy = 0.0
        minimum_elements_per_region = 1
        initial_clusters = :rand  # nothing in fact
        k = 1
    else
        maxiters = rand(space.maxiters)
        split_entropy = rand(space.split_entropy)
        minimum_elements_per_region = rand(space.minimum_elements_per_region)
        initial_clusters = rand(space.initial_clusters)
        k = rand(space.k)
    end

    config = KncConfig(
        rand(space.kernel),
        rand(space.centerselection),
        k,
        ncenters,
        maxiters,
        rand(space.recall),
        initial_clusters,
        split_entropy,
        minimum_elements_per_region
    )
end

"""
    combine_configurations(a::KncConfig, b::KncConfig)

Creates a new configuration combining the given configurations
"""
function combine_configurations(a::KncConfig, b::KncConfig)
    KncConfig(
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
