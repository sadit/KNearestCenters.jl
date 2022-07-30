# This file is a part of KNearestCenters.jl

export AbstractKernel, GaussianKernel, LaplacianKernel, CauchyKernel, SigmoidKernel, ReluKernel, TanhKernel, DirectKernel

abstract type AbstractKernel end

struct GaussianKernel <: AbstractKernel end
struct LaplacianKernel <: AbstractKernel end
struct CauchyKernel <: AbstractKernel end
struct SigmoidKernel <: AbstractKernel end
struct ReluKernel <: AbstractKernel end
struct TanhKernel <: AbstractKernel end
struct DirectKernel <: AbstractKernel end

"""
    kfun(kernel::GaussianKernel, d, σ::AbstractFloat)::Float64

Creates a Gaussian kernel with the given distance function
"""
function kfun(kernel::GaussianKernel, d, σ::AbstractFloat)::Float64
    d < 1e-6 ? 1.0 : exp(- d*d / (2 * σ * σ))
end

"""
    kfun(kernel::LaplacianKernel, d, σ::AbstractFloat)::Float64

Creates a Laplacian kernel with the given distance function
"""
function kfun(kernel::LaplacianKernel, d, σ::AbstractFloat)::Float64
    d < 1e-6 ? 1.0 : exp(- d / σ)
end

"""
    kfun(kernel::CauchyKernel, d, σ::AbstractFloat)::Float64

Creates a Cauchy kernel with the given distance function
"""
function kfun(kernel::CauchyKernel, d, σ::AbstractFloat)::Float64
    d < 1e-6 ? 1.0 : 1 / (1 + d*d / (σ * σ))
end

"""
    kfun(kernel::SigmoidKernel, d, σ::AbstractFloat)::Float64

Creates a Sigmoid kernel with the given distance function
"""
function kfun(kernel::SigmoidKernel, d, σ::AbstractFloat)::Float64
    d < 1e-6 ? 1.0 : 1 / (1 + exp(-1.0 + d/σ))
end

"""
    kfun(kernel::TanhKernel, d, σ::AbstractFloat)::Float64

Creates a Tanh kernel with the given distance function
"""
function kfun(kernel::TanhKernel, d, σ::AbstractFloat)::Float64
    d < 1e-6 && return 1.0
    x = σ - d
    (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

"""
    kfun(kernel::ReluKernel, d, σ::AbstractFloat)::Float64

Creates a Relu kernel with the given distance function
"""
function kfun(kernel::ReluKernel, d, σ::AbstractFloat)::Float64
    max(0.0, 1.0 - d / σ)
end

"""
    kfun(kernel::DirectKernel, d, σ::AbstractFloat)::Float64

Creates a Direct kernel with the given distance function
"""
function kfun(kernel::DirectKernel, d, σ::AbstractFloat)::Float64
    1 / (1e-5 + d)
end
