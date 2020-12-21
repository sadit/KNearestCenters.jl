# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export gaussian_kernel, laplacian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, relu_kernel, direct_kernel

"""
    gaussian_kernel(dist::Function)

Creates a Gaussian kernel with the given distance function
"""
function gaussian_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        d = dist(a, b)
        d < 1e-6 && return 1.0
        exp(- d*d / (2 * σ * σ))
    end
end

"""
    laplacian_kernel(dist::Function)

Creates a Laplacian kernel with the given distance function
"""
function laplacian_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        d = dist(a, b)
        d < 1e-6 && return 1.0
        exp(- d / σ)
    end
end

"""
    cauchy_kernel(dist::Function)

Creates a Laplacian kernel with the given distance function
"""
function cauchy_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        d = dist(a, b)
        d < 1e-6 && return 1.0
        1 / (1 + d*d / (σ * σ))
    end
end

"""
     sigmoid_kernel(dist::Function)

Creates a sigmoid kernel with the given `dist`
"""
function sigmoid_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        d = dist(a, b)
        d < 1e-6 && return 1.0
        1 / (1 + exp(-1.0 + d/σ))
    end
end

"""
    tanh_kernel(dist::Function)

Creates a tanh kernel with the given distance function
"""
function tanh_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        d = dist(a, b)
        d < 1e-6 && return 1.0
        x = σ - d
        (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    end
end

"""
    relu_kernel(dist::Function)

Creates a relu-like kernel with the given distance function 
"""
function relu_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        max(0.0, 1.0 - dist(a, b) / σ)
    end
end

"""
    direct_kernel(dist::Function)

Creates kernel just computing ``1/dist(\\cdot, \\cdot)``
"""
function direct_kernel(dist::Function)
    function fun(a, b, σ::AbstractFloat)::Float64
        1 / dist(a, b)
    end
end