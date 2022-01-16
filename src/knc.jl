# This file is a part of KNearestCenters.jl

@with_kw struct KncConfig{K_<:AbstractKernel, S_<:AbstractCenterSelection}
    kernel::K_ = ReluKernel(L2Distance())
    centerselection::S_ = CentroidSelection()
end

#KncConfig(nclasses; dist=L2Distance(), centerselection=CentroidSelection()) =
#    KncConfig(dist, centerselection)

@with_kw struct KncConfigSpace <: AbstractSolutionSpace
    kernel = [k_(d_()) for k_ in [DirectKernel, ReluKernel, GaussianKernel], d_ in [L2Distance, CosineDistance]]
    centerselection = [CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()]
end

Base.eltype(::KncConfigSpace) = KncConfig

"""
    rand(space::KncConfigSpace)

Creates a random `KncConfig` instance based on the `space` definition.
"""
function Base.rand(space::KncConfigSpace)
    #s = rand(0.8f0:0.01f0:1.0f0, space.nclasses)
    #s = ones(Float32, space.nclasses)
    KncConfig(rand(space.kernel), rand(space.centerselection))
end

"""
    combine(a::KncConfig, b::KncConfig)

Creates a new configuration combining the given configurations
"""
function combine(a::KncConfig, b::KncConfig)
    KncConfig(a.kernel, b.centerselection)
end

function mutate(space::KncConfigSpace, c::KncConfig, iter)
    combine(c, rand(space))
end

"""
A nearest centroid classifier with support for kernel functions
"""
struct Knc{DataType, K_<:KncConfig}
    config::K_
    centers::DataType
    dmax::Vector{Float32}
    res::KnnResult
end

"""
    Knc(config::KncConfig, X, y::CategoricalArray; verbose=true)

Creates a Knc classifier using the given configuration and data.
"""
function Knc(config::KncConfig, X, y::CategoricalArray; verbose=true)
    # computes a set of #labels centers using labels for clustering
    D = kcenters(config.kernel.dist, X, y, config.centerselection)
    @assert length(levels(y)) == length(D.centers)
    Knc(config, D.centers, D.dmax, KnnResult(1))
end

function predict(nc::Knc, x, res::KnnResult=reuse!(nc.res))
    C = nc.centers
    for i in eachindex(C)
        d = -evaluate(nc.config.kernel, x, C[i], nc.dmax[i])
        push!(res, i, d)
    end

    argmin(res)
end

function Base.broadcastable(nc::Knc)
    (nc,)
end