# This file is a part of KNearestCenters.jl

using Test, SimilaritySearch

function loadiris()
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = basename(url)
    if !isfile(filename)
        download(url, filename)
    end

    y = String[]
    X = Vector{Float32}[]
    for line in eachline(filename)
        arr = split(line, ',')
        length(arr) < 5 && continue
        vec = map(s->parse(Float32, s), arr[1:end-1])
        push!(X, vec)
        push!(y, arr[end])
    end

    VectorDatabase(X), y
end


function loadlinearreg()
    X = [rand(2) .+ i for i in 1:100]
    y = range(1, stop=100, length=100) .+ rand(100)
    X, y
end