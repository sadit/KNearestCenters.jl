# This file is a part of KNearestCenters.jl

using Test, SimilaritySearch, CSV

function loadiris()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url, filename)
    end

    X = CSV.read(filename, Tuple; header=0)
    permutedims(hcat(X[[1,2,3,4]]...)), X[end]
end

function loadlinearreg()
    X = [rand(2) .+ i for i in 1:100]
    y = range(1, stop=100, length=100) .+ rand(100)
    X, y
end