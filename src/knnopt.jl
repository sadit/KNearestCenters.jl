# This file is a part of KNearestCenters.jl

function optimize!(loss::Function, model::KnnModel, klist, wlist)
    n = length(model.index)
    best = Tuple[]
    for k in klist
        n < k && break
        model.k = k
        for w in wlist
            model.weight = w
            e = loss()
            push!(best, (e, k, w))
        end
    end

    sort!(best, by=first)
    m = first(best)
    _, model.k, model.weight = m
    # @show best
    model
end

function optimize!(
    model::KnnModel, loss::Loss, Xtest, ytest;
    klist=[1, 3, 5, 7, 11, 13, 15, 31],
    wlist=[KnnInvExpDistWeightKernel(),
           KnnUniformWeightKernel(), 
           KnnInvRankWeightKernel(),
           KnnPolyInvRankWeightKernel(),
           KnnInvDistWeightKernel()
        ]
    )
    optimize!(model, klist, wlist) do
        predict.(model, Xtest)
        value(loss, ytest, ypred)
    end
end

function optimize!(
        model::KnnModel, loss::Loss;
        klist=[1, 3, 5, 7, 11, 13, 15, 31],
        wlist=[
            KnnInvExpDistWeightKernel(),
            KnnUniformWeightKernel(), 
            KnnInvRankWeightKernel(),
            KnnPolyInvRankWeightKernel(),
            KnnInvDistWeightKernel()
        ],
        repeat=3,
        p=0.3
    )

    model.kstart = 2
    n = length(model.index)
    n > 0 || throw(InvalidSetupError(nothing, "invalid setup on optimize!, n=$n"))
    itestlist = [unique(rand(1:n, ceil(Int, n * p))) for _ in 1:repeat]

    optimize!(model, klist, wlist) do
        e = 0.0
        for itest in itestlist
            Xtest = model.index.db[itest]
            ytest = model.meta[itest]
            ypred = predict.(model, Xtest)
            e += value(loss, ytest, ypred)
        end

        e / repeat
    end

    model.kstart = 1
    model
end
