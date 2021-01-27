# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test
using KCenters, SimilaritySearch, KNearestCenters 
using CategoricalArrays, StatsBase, JSON

JSON.lower(f::Function) = string(f)

function generate_data()
    X = [rand(2) for i in 1:1000]
    y = categorical([round(Int, 1 + 2 * prod(x)) for x in X])
    X, y
end

@testset "AKNC" begin
    X, y = generate_data()
    X_, y_ = generate_data()

    #models = Dict()

    space = AKNC_ConfigSpace(
        ncenters=[0, 3, 7],
        k=[1],
        dist=[L2Distance, L1Distance],
        kernel=[DirectKernel],
        initial_clusters=[:fft, :rand, :dnet],
        minimum_elements_per_centroid=[1, 2]
    )
    best_list = search_params(space, X, y, 16;
        bsize=12,
        mutation_bsize=4,
        ssize=4,
        folds=3,
        tol=-1.0,
        search_maxiters=8,
        score=:accuracy,
        #models=models,
        verbose=true
    )
    @info "========== BEST MODEL =========="
    config, score = best_list[1]
    @test score > 0.9

    # @info get.(models[config], :model, nothing)
    A = AKNC(config, X, y)
    sa = classification_scores(y_.refs, predict.(A, X_))
    B = bagging(config, X, y, ratio=0.5, b=30)
    @test sa.accuracy > 0.80

    @info "class distribution: ", countmap(y), countmap(y_)
    @info "===== scores for single classifier: $(JSON.json(sa))"

    for k in [1, 5, 7, 9, 11]
        empty!(B.nc.res, k)
        sb = classification_scores(y_.refs, predict.(B, X_))
        @test sb.accuracy > 0.85
        @info "===== scores for $k: $(JSON.json(sb))"
    end

    # config_list = KNearestCenters.optimize!(B, X_, y_, accuracy_score; kernel=[DirectKernel], dist=[L1Distance, L2Distance])
    # sc = scores(y_, predict.(B, X_))
    # @test sc.accuracy > 0.85
    # @info "config: $(JSON.json(config_list[1]))"
    # @info "===== scores optimized! B: $(JSON.json(sc))"
end

