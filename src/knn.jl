# This file is a part of KNearestCenters.jl

using SimilaritySearch
using SparseArrays

export KnnModel, AbstractKnnPrediction, AbstractKnnWeightKernel
export KnnSingleLabelPrediction, KnnSoftmaxPrediction, KnnNormalizedPrediction
export KnnUniformWeightKernel, KnnInvRankWeightKernel, KnnPolyInvRankWeightKernel, KnnInvDistWeightKernel, KnnInvExpDistWeightKernel
export predict_raw, predict

function onehotenc(labels::CategoricalArray)
    I = Int32[]
    J = Int32[]
    W = Float32[]
    n = length(labels)
    sizehint!(I, n); sizehint!(J, n); sizehint!(W, n)
    L = levels(labels)
    m = length(L)
    
    for (i, l) in enumerate(labels)
        push!(I, levelcode(l))
        push!(J, i)
        push!(W, 1.0)
    end

    sparse(I, J, W, m, n), Dict(i => c for (i, c) in enumerate(L))
end

abstract type AbstractKnnPrediction end
abstract type AbstractKnnWeightKernel end

# Multi and single label kernels
struct KnnSingleLabelPrediction{MapType} <: AbstractKnnPrediction
    imap::MapType
end

struct KnnSoftmaxPrediction <: AbstractKnnPrediction end
Base.@kwdef struct KnnNormalizedPrediction <: AbstractKnnPrediction
    pnorm::Real = 2
end

# Weighting kernels
struct KnnUniformWeightKernel <: AbstractKnnWeightKernel end
struct KnnInvRankWeightKernel <: AbstractKnnWeightKernel end

Base.@kwdef struct KnnPolyInvRankWeightKernel <: AbstractKnnWeightKernel
    pow::Float32 = 2.0
end

Base.@kwdef struct KnnInvDistWeightKernel <: AbstractKnnWeightKernel
    eps::Float32 = 1e-3
end

Base.@kwdef struct KnnInvExpDistWeightKernel <: AbstractKnnWeightKernel
    num::Float32 = 1.0
    eps::Float32 = 1e-3
    pow::Float32 = 1.0
end

weight(::KnnUniformWeightKernel, d::Float32, rank::Int) = 1f0 
weight(::KnnInvRankWeightKernel, d::Float32, rank::Int) = 1f0 / rank
weight(kernel::KnnPolyInvRankWeightKernel, d::Float32, rank::Int) = 1f0 / rank^kernel.pow
weight(kernel::KnnInvDistWeightKernel, d::Float32, rank::Int) = 1f0 / (kernel.eps + d)
weight(kernel::KnnInvExpDistWeightKernel, d::Float32, rank::Int) = exp(kernel.num / (d + kernel.eps)^kernel.pow)


mutable struct KnnModel{PredictionType<:AbstractKnnPrediction, IndexType<:AbstractSearchIndex, MetaType<:AbstractArray}
    k::Int
    prediction::PredictionType
    weight::AbstractKnnWeightKernel
    index::IndexType
    meta::MetaType
end

function fit(::Type{KnnModel}, index::AbstractSearchIndex, meta::CategoricalArray; k=3, weight=KnnUniformWeightKernel())
    meta_, imap = onehotenc(meta)
    KnnModel(k, KnnSingleLabelPrediction(imap), weight, index, meta_)
end

function fit(::Type{KnnModel}, index::AbstractSearchIndex, meta::AbstractVecOrMat{<:Real}; k=3, weight=KnnUniformWeightKernel(), prediction=KnnSoftmaxPrediction())
    KnnModel(k, prediction, weight, index, meta)
end

function predict_(model::KnnModel, meta::AbstractSparseArray, res::KnnResult)
    pred = zeros(Float32, size(meta, 1))
    NZ = nonzeros(meta)
    RV = rowvals(meta)

    for (rank, (id, dist)) in enumerate(res)
        w = weight(model.weight, dist, rank)

        for i in nzrange(meta, id)
            pred[RV[i]] = w * NZ[i]
        end        
    end

    pred
end

function predict_(model::KnnModel, meta::DenseArray, res::KnnResult)
    m = size(meta, 1)
    pred = zeros(Float32, m)

    for (rank, (id, dist)) in enumerate(res)
        w = weight(model.weight, dist, rank)
        V = view(meta, :, id)

        @inbounds for i in eachindex(pred)
            pred[i] += w * V[i]
        end
    end

    pred
end

function predict_raw(model::KnnModel, x)
    res = getknnresult(model.k)
    search(model.index, x, res)
    predict_(model, model.meta, res)
end

normalize_by_kind!(::KnnSoftmaxPrediction, pred) = softmax!(pred)
normalize_by_kind!(p::KnnNormalizedPrediction, pred) = normalize!(pred, p.pnorm)

function normalize_by_kind!(m::KnnSingleLabelPrediction, pred)
    _, p = findmax(pred)
    m.imap[p]
end

function predict(mode::KnnModel, x)
    normalize_by_kind!(mode.prediction, predict_raw(mode, x))
end

#=function optimize!()
end=#