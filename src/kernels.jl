# This file is a part of KNearestCenters.jl

export AbstractKernel, GaussianKernel, LaplacianKernel, CauchyKernel, SigmoidKernel, ReluKernel, TanhKernel, DirectKernel
import SimilaritySearch: evaluate, SemiMetric

abstract type AbstractKernel end

struct GaussianKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct LaplacianKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct CauchyKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct SigmoidKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct ReluKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct TanhKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end

struct DirectKernel{DistType<:SemiMetric} <: AbstractKernel
    dist::DistType
end


"""
    evaluate(kernel::GaussianKernel, a, b, σ::AbstractFloat)::Float64

Creates a Gaussian kernel with the given distance function
"""
function evaluate(kernel::GaussianKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    d < 1e-6 ? 1.0 : exp(- d*d / (2 * σ * σ))
end

"""
    evaluate(kernel::LaplacianKernel, a, b, σ::AbstractFloat)::Float64

Creates a Laplacian kernel with the given distance function
"""
function evaluate(kernel::LaplacianKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    d < 1e-6 ? 1.0 : exp(- d / σ)
end

"""
    evaluate(kernel::CauchyKernel, a, b, σ::AbstractFloat)::Float64

Creates a Cauchy kernel with the given distance function
"""
function evaluate(kernel::CauchyKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    d < 1e-6 ? 1.0 : 1 / (1 + d*d / (σ * σ))
end

"""
    evaluate(kernel::SigmoidKernel, a, b, σ::AbstractFloat)::Float64

Creates a Sigmoid kernel with the given distance function
"""
function evaluate(kernel::SigmoidKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    d < 1e-6 ? 1.0 : 1 / (1 + exp(-1.0 + d/σ))
end

"""
    evaluate(kernel::TanhKernel, a, b, σ::AbstractFloat)::Float64

Creates a Tanh kernel with the given distance function
"""
function evaluate(kernel::TanhKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    d < 1e-6 && return 1.0
    x = σ - d
    (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

"""
    evaluate(kernel::ReluKernel, a, b, σ::AbstractFloat)::Float64

Creates a Relu kernel with the given distance function
"""
function evaluate(kernel::ReluKernel, a, b, σ::AbstractFloat)::Float64
    d = evaluate(kernel.dist, a, b)
    max(0.0, 1.0 - d / σ)
end

"""
    evaluate(kernel::DirectKernel, a, b, σ::AbstractFloat)::Float64

Creates a Direct kernel with the given distance function
"""
function evaluate(kernel::DirectKernel, a, b, σ::AbstractFloat)::Float64
    1 / evaluate(kernel.dist, a, b)
end
