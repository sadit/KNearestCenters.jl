# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

include("loaddata.jl")
using KCenters, SimilaritySearch
using StatsBase, CategoricalArrays

@testset "One class classifier with DeloneHistogram" begin

    X, ylabels = loadiris()
    dist = L2Distance()

    for k in [3, 5, 7, 11]
        println("===> k=$k")
        for initial in [:fft, :dnet, :rand], maxiters in [0, 3, 10]
            L = Float64[]
            for label in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
                y = ylabels .== label
                A = X[y]
                centers = kcenters(dist, A, k, initial=initial, maxiters=maxiters, tol=0.0)
                vor = DeloneHistogram(dist, centers)
                ypred = [predict(vor, q) for q in X]
                push!(L, mean(ypred .== y))
                @test L[end] > 0.5
            end

            macrorecall = mean(L)
            println(stderr, "===> (k=$k, initial=$initial, maxiters=$maxiters); macro-recall: $macrorecall")
            @test macrorecall > 0.7
        end
    end
end


@testset "KNC" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = categorical([M[y] for y in ylabels])
    dist = L2Distance()
    C = kcenters(dist, X, 12)
    summary = most_frequent_label
    for kernel in [GaussianKernel(dist), LaplacianKernel(dist), CauchyKernel(dist), SigmoidKernel(dist), TanhKernel(dist), ReluKernel(dist), DirectKernel(dist)]
        @info "XXXXXX==== split_entropy>", kernel
        nc = KNC(kernel, C, X, y, verbose=true, split_entropy=0.5)
        @show nc.class_map
        ypred = [predict(nc, x) for x in X]
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8
        
        break
    end

    C = kcenters(dist, X, 12, maxiters=0, verbose=true)
    nc = KNC(DirectKernel(dist), C, X, y, verbose=true, split_entropy=0.5)
    empty!(nc.res, 3) ## using 3 nearest centers instead of one
    ypred = [predict(nc, x) for x in X] # a direct kernel is required for the iris dataset and knn
    acc = mean(ypred .== y)
    @show acc
    @test acc > 0.8

end

