# This file is a part of KNearestCenters.jl

using Test

include("loaddata.jl")
using KCenters, SearchModels, SimilaritySearch
using Random, StatsBase, CategoricalArrays, MLDataUtils

@testset "NearestCenter search_models" begin
    X, ylabels = loadiris()
    ylabels = categorical(ylabels)
    space = KncConfigSpace()
    Random.seed!(2)
    ifolds = kfolds(shuffle!(collect(1:length(X))), 3)
    function errorfun(config::KncConfig)
        err = 0.0
        for (itrain, itest) in ifolds
            model = Knc(config, X[itrain], ylabels[itrain])
            yhat = predict.(model, X[itest])
            err += mean(yhat .== ylabels[itest].refs)
        end
    
        1.0 - err / length(ifolds)
    end

    best_list = search_models(space, errorfun, 16;
        bsize=8,
        mutbsize=4,
        crossbsize=8,
        tol=-1.0,
        maxiters=16,
        verbose=true
    )

    B = best_list[1]
    println(stderr, "=== BEST MODEL:", B)
    @test 1 - B.second > 0.9
end