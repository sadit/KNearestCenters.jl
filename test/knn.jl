# This file is a part of KNearestCenters.jl

using Test

include("loaddata.jl")
using SimilaritySearch, KNearestCenters, Random
using StatsBase: mean
using Random

@testset "NearestCenter search_models" begin
    Xtrain, ytrain, Xtest, ytest = loadiris()
    model = fit(KnnModel, Xtrain, categorical(ytrain))
    m = optimize!(model, BalancedErrorRate(), Xtest, ytest)
    ypred = predict.(model, Xtest)
    @test 0.8 < recall_score(ytest, ypred)
end