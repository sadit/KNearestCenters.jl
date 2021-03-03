# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

@with_kw struct KncConfig{K_<:AbstractKernel, S_<:AbstractCenterSelection}
    kernel::K_ = ReluKernel(L2Distance())
    centerselection::S_ = CentroidSelection()
end

#KncConfig(nclasses; dist=L2Distance(), centerselection=CentroidSelection()) =
#    KncConfig(dist, centerselection)

StructTypes.StructType(::Type{<:KncConfig}) = StructTypes.Struct()

@with_kw struct KncConfigSpace <: AbstractSolutionSpace
    kernel = [k_(d_()) for k_ in [DirectKernel, ReluKernel, GaussianKernel], d_ in [L2Distance, CosineDistance]]
    centerselection = [CentroidSelection(), RandomCenterSelection(), MedoidSelection(), KnnCentroidSelection()]
end

Base.eltype(::KncConfigSpace) = KncConfig

"""
    random_configuration(space::KncConfigSpace)

Creates a random `KncConfig` instance based on the `space` definition.
"""
function random_configuration(space::KncConfigSpace)
    #s = rand(0.8f0:0.01f0:1.0f0, space.nclasses)
    #s = ones(Float32, space.nclasses)
    KncConfig(rand(space.kernel), rand(space.centerselection))
end

"""
    combine_configurations(a::KncConfig, b::KncConfig)

Creates a new configuration combining the given configurations
"""
function combine_configurations(a::KncConfig, b::KncConfig)
    L = [a, b]
    # m = (a.scale .+ b.scale) ./ 2
    KncConfig(rand(L).kernel, rand(L).centerselection)
end

"""
A nearest centroid classifier with support for kernel functions
"""
struct Knc{DataType<:AbstractVector, K_<:KncConfig}
    config::K_
    centers::DataType
    dmax::Vector{Float32}
    res::KnnResult
end

StructTypes.StructType(::Type{<:Knc}) = StructTypes.Struct()

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

function predict(nc::Knc, x, res::KnnResult=nc.res)
    empty!(nc.res)
    C = nc.centers
    for i in eachindex(C)
        d = -evaluate(nc.config.kernel, x, C[i], nc.dmax[i])
        push!(res, i, d)
    end

    first(res).id
end

function Base.broadcastable(nc::Knc)
    (nc,)
end