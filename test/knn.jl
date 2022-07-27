# This file is a part of KNearestCenters.jl

using Test

include("loaddata.jl")
using SimilaritySearch, KNearestCenters, Random, MLUtils
using StatsBase: mean
using Random


@testset "NearestCenter search_models" begin
    X, ylabels = loadiris()
    itrain, itest = splitobs(1:length(ylabels), at=0.5, shuffle=true)
    Xtrain, ytrain = MatrixDatabase(X[:, itrain]), ylabels[itrain]
    Xtest, ytest = MatrixDatabase(X[:, itest]), ylabels[itest]
    model = fit(KnnModel, Xtrain, categorical(ytrain))
    m = optimize!(model, BalancedErrorRate(), Xtest, ytest)
    ypred = predict.(model, Xtest)
    @show m.k, m.weight, recall_score(ytest, ypred)
end