# This file is a part of KNearestCenters.jl

Base.@kwdef struct KncConfig{K_<:AbstractKernel, D_<:SemiMetric, S_<:AbstractCenterSelection}
    kernel::K_ = ReluKernel()
    dist::D_ = L2Distance()
    centerselection::S_ = CentroidSelection()
end

Base.@kwdef struct KncConfigSpace <: AbstractSolutionSpace
    kernel = [DirectKernel(), ReluKernel(), GaussianKernel()]
    dist = [L1Distance(), L2Distance(), CosineDistance()]
    centerselection = [CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()]
end

Base.eltype(::KncConfigSpace) = KncConfig

"""
    rand(space::KncConfigSpace)

Creates a random `KncConfig` instance based on the `space` definition.
"""
function Base.rand(space::KncConfigSpace)
    KncConfig(rand(space.kernel), rand(space.dist), rand(space.centerselection))
end

"""
    combine(a::KncConfig, b::KncConfig)

Creates a new configuration combining the given configurations
"""
function combine(a::KncConfig, b::KncConfig)
    KncConfig(
        rand((a.kernel, b.kernel)),
        rand((a.dist, b.dist)),
        rand((a.centerselection, b.centerselection))
    )
end

function mutate(space::KncConfigSpace, c::KncConfig, iter)
    combine(c, rand(space))
end

"""
A nearest centroid classifier with support for kernel functions
"""
struct Knc{DataType,K_<:KncConfig,D_}
    config::K_
    centers::DataType
    dmax::Vector{Float32}
    imap::D_
end

"""
    fit(config::KncConfig, X, y::CategoricalArray; verbose=true)

Creates a Knc classifier using the given configuration and data.
"""
function fit(config::KncConfig, X::AbstractDatabase, y::CategoricalArray; verbose=true)
    # computes a set of #labels centers using labels for clustering
    D = kcenters(config.dist, X, y, config.centerselection)
    @assert length(levels(y)) == length(D.centers)
    Knc(config, D.centers, D.dmax, Dict(i => c for (i, c) in enumerate(levels(y))))
end

function predict(nc::Knc, x)
    C, kernel, dist = nc.centers, nc.config.kernel, nc.config.dist
    res = KnnResult(1)
    for i in eachindex(C)
        d = kfun(kernel, evaluate(dist, x, C[i]), nc.dmax[i])
        push_item!(res, i, -d)
    end

    nc.imap[argmin(res)]
end

function Base.broadcastable(nc::Knc)
    (nc,)
end
