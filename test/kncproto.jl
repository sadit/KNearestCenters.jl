# This file is a part of KNearestCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test

include("loaddata.jl")
using KCenters, SearchModels, SimilaritySearch
using Random, StatsBase, CategoricalArrays, MLDataUtils, JSON3
Random.seed!(1)

@testset "KncProto" begin
    X, ylabels = loadiris()
    M = Dict(label => i for (i, label) in enumerate(unique(ylabels) |> sort!))
    y = categorical([M[y] for y in ylabels])
    dist = L2Distance()
    C = kcenters(dist, X, 12)
    for kernel in [GaussianKernel(dist), LaplacianKernel(dist), CauchyKernel(dist), SigmoidKernel(dist), TanhKernel(dist), ReluKernel(dist), DirectKernel(dist)]
        nc = KncProto(KncProtoConfig(kernel=kernel, ncenters=-7), X, y, verbose=true)
        ypred = predict.(nc, X)
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8

        nc = KncProto(KncProtoConfig(kernel=kernel, ncenters=21, initial_clusters=:fft, minimum_elements_per_region=1), X, y, verbose=true)
        ypred = predict.(nc, X)
        acc = mean(ypred .== y)
        @show acc
        @test acc > 0.8
    end
end

@testset "KncPerClass search_models" begin
    X, ylabels = loadiris()
    ylabels = categorical(ylabels)
    space = KncPerClassConfigSpace{0.3}(
        initial_clusters=[:rand]
    )
    
    ifolds = kfolds(shuffle!(collect(1:length(X))), 3)
    function errfun(config::KncProtoConfig)
        err = 0.0
        for (itrain, itest) in ifolds
            model = KncProto(config, X[itrain], ylabels[itrain])
            err += mean(predict.(model, X[itest]) .== ylabels[itest].refs)
        end
    
        err = 1.0 - err / length(ifolds)
        println(stderr, err, "\t", typeof(config), "\t", JSON3.write(config))
        err    
    end

    best_list = search_models(space, errfun, 32;
        bsize=8,
        mutbsize=8,
        crossbsize=8,
        tol=-1.0,
        maxiters=4,
        verbose=true
    )
    @info "========== BEST MODEL ==========", best_list[1]
    @test 1 - best_list[1].second > 0.9
end


@testset "KncProto search_models" begin
    X, ylabels = loadiris()
    ylabels = categorical(ylabels)
    space = KncGlobalConfigSpace{0.3}(
        initial_clusters=[:rand]
    )
    
    ifolds = kfolds(shuffle!(collect(1:length(X))), 3)
    function errfun(config::KncProtoConfig)
        err = 0.0
        for (itrain, itest) in ifolds
            model = KncProto(config, X[itrain], ylabels[itrain])
            err += mean(predict.(model, X[itest]) .== ylabels[itest].refs)
        end
    
        err = 1.0 - err / length(ifolds)
        println(stderr, err, "\t", typeof(config), "\t", JSON3.write(config))
        err    
    end

    best_list = search_models(space, errfun, 32;
        bsize=8,
        mutbsize=8,
        crossbsize=8,
        tol=-1.0,
        maxiters=4,
        verbose=true
    )
    @info "========== BEST MODEL ==========", best_list[1]
    @test 1 - best_list[1].second > 0.9
end