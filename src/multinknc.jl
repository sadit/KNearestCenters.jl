# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export glue, bagging, optimize!

"""
    glue(arr::AbstractVector{K}) where K <: KNC
    glue(arr::AbstractVector{K}) where K <: AKNC

In its first form it joins a list of KNC classifiers into a single one.
The second form, it also joins the classifiers into a single AKNC classifier;
however, it uses takes specifications (the kernel function and the configuration)
of the first element of the list.

"""
function glue(arr::AbstractVector{K}) where K <: KNC
    centers = vcat([c.centers for c in arr]...)
    dmax = vcat([c.dmax for c in arr]...)
    class_map = vcat([c.class_map for c in arr]...)
    KNC(centers, dmax, class_map, arr[1].nclasses)
end

function glue(arr::AbstractVector{K}) where K <: AKNC
    a = first(arr)
    nc = glue([s.nc for s in arr])
    AKNC(nc, a.kernel, a.config)
end

"""
    bagging(config::AKNC_Config, X::AbstractVector, y::AbstractVector{I}; b=13, ratio=0.5) where {I<:Integer}

Creates `b` classifiers, each trained with a random `ratio` of the dataset;
these classifiers are joint into a single classifier with `glue`.
"""
function bagging(config::AKNC_Config, X::AbstractVector, y::CategoricalArray; b=13, ratio=0.5)
    indexes = collect(1:length(X))
    m = ceil(Int, ratio * length(X))

    L = Vector{AKNC}(undef, b)
    for i in 1:b
        shuffle!(indexes)
        sample = @view indexes[1:m]
        L[i] = fit(AKNC, config, X[sample], y[sample])
    end

    glue(L)
end

"""
    optimize!(model::AKNC, X, y, score::Function=recall_score; k=[1, 3, 5, 7], kernel=[direct_kernel, relu_kernel, laplacian_kernel, gaussian_kernel], dist=[l1_distance, l2_distance, full_angle_distance], summary=[most_frequent_label, mean_label], verbose=true)

Selects `k` and `kernel` to AKNC to adjust better to the given score and the dataset ``(X, y)``.
"""
function optimize!(model::AKNC, X, y, score::Function=recall_score; k=[1, 3, 5, 7], kernel=[direct_kernel, relu_kernel, laplacian_kernel, gaussian_kernel], dist=[l1_distance, l2_distance, full_angle_distance], summary=[most_frequent_label, mean_label], verbose=true)
    L = []
    for k_ in k, kernel_ in kernel, dist_ in dist, summary_ in summary
        kernel_fun = kernel_(dist_)
        model.config.k = k_
        model.config.summary = summary_
        model.kernel = kernel_fun
        ypred = predict(model, X)
        s = score(y, ypred)
        push!(L, (score=s, k=k_, kernel=kernel_, dist=dist_, summary=summary_, kernel_fun=kernel_fun))
        verbose && println(stderr, L[end])
    end

    sort!(L, by=x->x.score, rev=true)
    c = first(L)
    model.config.k = c.k
    model.config.summary = c.summary
    model.config.kernel = c.kernel
    model.config.dist = c.dist
    model.kernel = c.kernel_fun
    L
end
