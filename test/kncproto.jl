# This file is a part of KNearestCenters.jl

using Test

include("loaddata.jl")
using KCenters, SearchModels, SimilaritySearch, Random, MLUtils
using StatsBase: mean
Random.seed!(1)

@testset "KncProto" begin
    Xtrain, ytrain, Xtest, ytest = loadiris()
    nc = fit(KncProtoConfig(ncenters=7), Xtrain, ytrain, verbose=true)
    ypred = predict.(nc, Xtest)
    acc = mean(ypred .== ytest)
    @test acc > 0.8

    nc = fit(KncProtoConfig(ncenters=17), Xtrain, ytrain, verbose=true)
    ypred = predict.(nc, Xtest)
    acc12 = mean(ypred .== ytest)
    @test acc12 > 0.8

    @show acc, acc12
end

@testset "KncProto search_models" begin
    Xtrain, ytrain, Xtest, ytest = loadiris()
    space = KncProtoConfigSpace{0.3}(
        initial_clusters=[:rand]
    )
    
    n = length(ytrain)
    ifolds = kfolds(n, 3)
    function errfun(config::KncProtoConfig)
        err = 0.0
        for (itrain, itest) in ifolds
            model = fit(config, Xtrain[itrain], ytrain[itrain])
            err += mean(predict.(model, Xtrain[itest]) .== ytrain[itest])
        end
    
        1.0 - err / length(ifolds)
    end

    best_list = search_models(errfun, space, 32, SearchParams(
            bsize=8,
            mutbsize=8,
            crossbsize=8,
            maxiters=4,
            verbose=true
        )
    )
    @info "========== BEST MODEL ==========", best_list[1]
    @test 1 - best_list[1].second > 0.9

    nc = fit(best_list[1].first, Xtrain, ytrain, verbose=true)
    ypred = predict.(nc, Xtest)
    acc = mean(ypred .== ytest)
    @test acc > 0.8
    @show acc
end
