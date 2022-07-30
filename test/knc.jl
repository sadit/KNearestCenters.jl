# This file is a part of KNearestCenters.jl

using Test

include("loaddata.jl")
using KCenters, SearchModels, SimilaritySearch, MLUtils
using StatsBase: mean
using Random


@testset "NearestCenter search_models" begin
    Xtrain, ytrain, Xtest, ytest = loadiris(at=0.7)
    space = KncConfigSpace()
    Random.seed!(2)
    n = length(ytrain)
    ifolds = kfolds(n, 3)
    function errorfun(config::KncConfig)
        err = 0.0
        for (itrain, itest) in ifolds
            model = fit(config, Xtrain[itrain], ytrain[itrain])
            yhat = predict.(model, Xtrain[itest])
            err += mean(yhat .== ytrain[itest])
        end
    
        1.0 - err / length(ifolds)
    end

    best_list = search_models(errorfun, space, 16, SearchParams(
            bsize=8,
            mutbsize=4,
            crossbsize=8,
            tol=-1.0,
            maxiters=16,
            verbose=true
        )
    )

    B = best_list[1]
    println(stderr, "=== BEST MODEL:", B)
    model = fit(B[1], Xtrain, ytrain)
    ypred = predict.(model, Xtest)
    @test 0.8 < recall_score(ytest, ypred)
end