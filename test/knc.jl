# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

include("loaddata.jl")
using KCenters, SearchModels, SimilaritySearch
using Random, StatsBase, CategoricalArrays, MLDataUtils


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


@testset "Knc" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = categorical([M[y] for y in ylabels])
    dist = L2Distance()
    C = kcenters(dist, X, 12)
    for kernel in [GaussianKernel(dist), LaplacianKernel(dist), CauchyKernel(dist), SigmoidKernel(dist), TanhKernel(dist), ReluKernel(dist), DirectKernel(dist)]
        nc = Knc(KncConfig(kernel=kernel, ncenters=0), X, y, verbose=true)
        ypred = predict.(nc, X)
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8

        nc = Knc(KncConfig(kernel=kernel, ncenters=21, initial_clusters=:fft, minimum_elements_per_region=1), X, y, verbose=true)
        ypred = predict.(nc, X)
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8
    end
end

@testset "Knc search_models" begin
    X, ylabels = loadiris()
    ylabels = categorical(ylabels)
    space = KncConfigSpace(
        ncenters=[-7, -3, 0, 3, 7],
        k=[1],
        initial_clusters=[:fft, :dnet],
        minimum_elements_per_region=[3]
    )
    
    ifolds = kfolds(shuffle!(collect(1:length(X))), 3)
    function evalmodel(config::KncConfig)
        score = 0.0
        for (itrain, itest) in ifolds
            model = Knc(config, X[itrain], ylabels[itrain])
            score += mean(predict.(model, X[itest]) .== ylabels[itest].refs)
        end
    
        -score / length(ifolds)
    end

    best_list = search_models(space, evalmodel, 32;
        bsize=8,
        mutbsize=8,
        crossbsize=8,
        tol=-1.0,
        maxiters=4,
        verbose=true
    )
    @info "========== BEST MODEL ==========", best_list[1]
    @test abs(best_list[1].second) > 0.9
end