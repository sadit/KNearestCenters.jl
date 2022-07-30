# This file is a part of KNearestCenters.jl

using Test, SimilaritySearch, MLUtils, CSV

function loadiris(; at=0.5, shuffle=true)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url, filename)
    end

    X = CSV.read(filename, Tuple; header=0)
    X, ylabels = permutedims(hcat(X[[1,2,3,4]]...)), X[end]

    itrain, itest = splitobs(1:length(ylabels); at, shuffle)
    Xtrain, ytrain = MatrixDatabase(X[:, itrain]), categorical(ylabels[itrain])
    Xtest, ytest = MatrixDatabase(X[:, itest]), ylabels[itest]
    Xtrain, ytrain, Xtest, ytest
end

function loadlinearreg()
    X = [rand(2) .+ i for i in 1:100]
    y = range(1, stop=100, length=100) .+ rand(100)
    X, y
end