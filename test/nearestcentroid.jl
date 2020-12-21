# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

include("loaddata.jl")
using KCenters, SimilaritySearch
using StatsBase, CategoricalArrays

@testset "One class classifier with DeloneHistogram" begin

    X, ylabels = loadiris()
    dist = l2_distance

    for k in [3, 5, 7, 11]
        println("===> k=$k")
        for initial in [:fft, :dnet, :rand], maxiters in [0, 3, 10]
            L = Float64[]
            for label in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
                y = ylabels .== label
                A = X[y]
                centers = kcenters(dist, A, k, initial=initial, maxiters=maxiters, tol=0.0)
                vor = fit(DeloneHistogram, centers)
                ypred = predict(vor, dist, X)
                push!(L, mean(ypred .== y))
                @test L[end] > 0.6
            end

            macrorecall = mean(L)
            println(stderr, "===> (k=$k, initial=$initial, maxiters=$maxiters); macro-recall: $macrorecall")
            @test macrorecall > 0.8
        end
    end
end

@testset "KNC" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = categorical([M[y] for y in ylabels])
    dist = l2_distance
    C = kcenters(dist, X, 12)
    summary = most_frequent_label
    for kernel in [gaussian_kernel, laplacian_kernel, cauchy_kernel, sigmoid_kernel, tanh_kernel, relu_kernel, direct_kernel]
        @info "XXXXXX==== split_entropy>", (kernel, dist)
        nc = fit(KNC, dist, C, X, y, verbose=true, split_entropy=0.5)
        @show nc.class_map
        ypred = predict(nc, kernel(dist), summary, X, 1)
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8
        
        
    end

    C = kcenters(dist, X, 12, maxiters=0, verbose=true)
    nc = fit(KNC, dist, C, X, y, verbose=true, split_entropy=0.5)
    ypred = predict(nc, direct_kernel(dist), summary, X, 3) # a direct kernel is required for the iris dataset and knn
    acc = mean(ypred .== y)
    @show acc
    @test acc > 0.8

end

