# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using KCenters
using MLDataUtils, Distributed, Random, StatsBase
import StatsBase: fit, predict
import Base: hash, isequal
export search_params, random_configurations, combine_configurations, fit, after_load, predict, AKNC, AKNC_Config
import Base: hash, isequal

mutable struct AKNC_Config
    kernel::Function
    dist::Function
    centroid::Function
    summary::Function

    k::Int
    ncenters::Int
    maxiters::Int
    
    recall::Float64
    initial_clusters
    split_entropy::Float64
    minimum_elements_per_centroid::Int
end

function AKNC_Config(;
        kernel::Function=relu_kernel, # [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel]
        dist::Function=l2_distance,
        centroid::Function=mean,
        summary::Function=most_frequent_label,

        k::Int=1,
        ncenters::Integer=0,
        maxiters::Integer=1,
        
        recall::AbstractFloat=1.0,
        initial_clusters=:rand,
        split_entropy::AbstractFloat=0.6,
        minimum_elements_per_centroid=3)
    
    AKNC_Config(
        kernel, dist, centroid, summary,
        k, ncenters, maxiters,
        recall, initial_clusters, split_entropy, minimum_elements_per_centroid)
end

hash(a::AKNC_Config) = hash(repr(a))
isequal(a::AKNC_Config, b::AKNC_Config) = isequal(repr(a), repr(b))

mutable struct AKNC{T}
    nc::KNC{T}
    kernel::Function
    config::AKNC_Config
end

"""
    fit(::Type{AKNC}, config::AKNC_Config, X, y; verbose=true)
    fit(config::AKNC_Config, X, y; verbose=true)

Creates a new `AKNC` model using the given configuration and the dataset `X` and `y`
"""
function fit(::Type{AKNC}, config::AKNC_Config, X, y; verbose=true)    
    if config.ncenters == 0
        C = kcenters(config.dist, X, y, config.centroid)
        cls = fit(KNC, C)
    else
        C = kcenters(config.dist, X, config.ncenters, config.centroid,
            initial=config.initial_clusters, recall=config.recall, verbose=verbose, maxiters=config.maxiters)
        cls = fit(
            KNC, cosine_distance, C, X, y,
            config.centroid,
            split_entropy=config.split_entropy,
            minimum_elements_per_centroid=config.minimum_elements_per_centroid,
            verbose=verbose)
    end

    AKNC(cls, config.kernel(config.dist), config)
end

fit(config::AKNC_Config, X, y; verbose=true) = fit(AKNC, config, X, y; verbose=verbose)

"""
    predict(model::AKNC, X, k::Integer=0)

Predicts the label of each item in `X` using `model`; k == 0 means for using the stored `k` in `config`
"""

function predict(model::AKNC, X, k::Integer=0)
    k = k == 0 ? model.config.k : k
    ypred = predict(model.nc, model.kernel, model.config.summary, X, k)
end
"""
    after_load(model::AKNC)

Fixes the `AKNC` after loading it from an stored image. In particular, it creates a function composition among distance function and a non-linear function with specific properties. 
"""
function after_load(model::AKNC)
    model.kernel = model.config.kernel(config.dist)
end

"""
    evaluate_model(config::AKNC_Config, train_X, train_y, test_X, test_y; verbose=true)

Creates a model for `train_X` and `train_y`, defined with `config`, evaluates them with `test_X` and `test_y`.
Returns a named tuple containing the evalution scores and the computed model.
"""
function evaluate_model(config::AKNC_Config, train_X, train_y, test_X, test_y; verbose=true)
    knc = fit(AKNC, config, train_X, train_y, verbose=verbose)
    ypred = predict(knc, test_X)
    (scores=scores(test_y, ypred), model=knc)
end

"""
    random_configurations(::Type{AKNC}, H, ssize;
        kernel::AbstractVector=[relu_kernel], 
        dist::AbstractVector=[l2_distance],
        k::AbstractVector=[1],
        maxiters::AbstractVector=[1, 3, 10],
        recall::AbstractVector=[1.0],
        ncenters::AbstractVector=[0, 10],
        initial_clusters::AbstractVector=[:fft, :dnet, :rand],
        split_entropy::AbstractVector=[0.3, 0.6, 0.9],
        minimum_elements_per_centroid::AbstractVector=[1, 3, 5],
        verbose=true
    )

Creates `ssize` random configurations for AKNC (they will be stored in the `H` dictionary) using
the search space definition of the given parameters (the following parameters must be given as vectors of possible choices)

- `kernel` a kernel function [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel, direct_kernel], see `src/kernels.jl`
- `dist` function to measure the distance between any two valid objects
- `k` is number of nearest centroids to determine labels,
- `maxiters` is number of iterations of the Lloyd's algorithm for computing clusters
- `recall` determines if an approximate metric index should be used for computing cluster, `0 < recall \\leq 1`, it trades quality by speed.
- `ncenters` number of centers to compute (0 means to use the labeled data to compute clusters)
- `initial_clusters` specifies how to compute initial clusters [:fft, :dnet, :rand]; also, an actual array of clusters can be given.
- `split_entropy` determines when a cluster can be splitted; i.e., when the entropy of the cluster surpasses this threshold, only for ``ncenters> 0``.
- `minimum_elements_per_centroid`, the algorithm will refuse to create clusters with less than this number of elements, only for ``ncenters> 0``.
- `verbose` controls the verbosity of the output

"""
function random_configurations(::Type{AKNC}, H, ssize;
        kernel::AbstractVector=[relu_kernel, direct_kernel], # [gaussian_kernel, laplacian_kernel, sigmoid_kernel, relu_kernel]
        dist::AbstractVector=[l2_distance],
        centroid::AbstractVector=[mean],
        summary::AbstractVector=[most_frequent_label, mean_label],
        k::AbstractVector=[1],
        maxiters::AbstractVector=[1, 3, 10],
        recall::AbstractVector=[1.0],
        ncenters::AbstractVector=[0, 10],
        initial_clusters::AbstractVector=[:fft, :dnet, :rand],
        split_entropy::AbstractVector=[0.3, 0.6, 0.9],
        minimum_elements_per_centroid::AbstractVector=[1, 3, 5],
        verbose=true
    )

    _rand_list(lst) = length(lst) == 0 ? [] : rand(lst)

    H = H === nothing ? Dict{AKNC_Config,Float64}() : H
    iter = 0
    for i in 1:ssize
        iter += 1
        ncenters_ = rand(ncenters)
        if ncenters_ == 0
            maxiters_ = 0
            split_entropy_ = 0.0
            minimum_elements_per_centroid_ = 1
            initial_clusters_ = :rand # nothing in fact
            k_ = 1
        else
            maxiters_ = rand(maxiters)
            split_entropy_ = rand(split_entropy)
            minimum_elements_per_centroid_ = rand(minimum_elements_per_centroid)
            initial_clusters_ = rand(initial_clusters)
            k_ = rand(k)
        end

        config = AKNC_Config(
            kernel = rand(kernel),
            dist = rand(dist),
            centroid = rand(centroid),
            summary = rand(summary),
            k = k_,
            ncenters = ncenters_,
            maxiters = maxiters_,
            recall = rand(recall),
            initial_clusters = initial_clusters_,
            split_entropy = split_entropy_,
            minimum_elements_per_centroid = minimum_elements_per_centroid_
        )
        haskey(H, config) && continue
        H[config] = -1
    end

    H
end

"""
    combine_configurations(config_list::AbstractVector{AKNC_Config}, ssize, H)

Creates `ssize` individuals using a combination of the given `config_list` (they will be stored in the `H` dictionary)
"""
function combine_configurations(config_list::AbstractVector{AKNC_Config}, ssize, H)
    function _sel()
        rand(config_list)
    end

    a = _sel()
    for i in 1:ssize
        config = AKNC_Config(
            kernel = _sel().kernel,
            dist = _sel().dist,
            centroid = _sel().centroid,
            summary = _sel().summary,
            k = a.k,
            ncenters = a.ncenters,
            maxiters = a.maxiters,
            recall = _sel().recall,
            initial_clusters = a.initial_clusters,
            split_entropy = a.split_entropy,
            minimum_elements_per_centroid = a.minimum_elements_per_centroid,
        )
        haskey(H, config) && continue
        H[config] = -1
    end

    H
end

"""
    search_params(::Type{AKNC}, X, y, configurations;
        bsize::Integer=4,
        mutation_bsize::Integer=1,
        ssize::Integer=8,
        folds=0.7,
        search_maxiters::Integer=8,
        score=:macro_recall,
        tol::AbstractFloat=0.01,
        verbose=true,
        models::Union{Nothing,Dict}=nothing,
        distributed=true,
        config_kwargs...
    )

Performs a model selection of AKNC for the given examples ``(X, y)`` using an evolutive algorithm
(a variant of a genetic algorithm). Note that this function use Distributed's `@spawn` to evaluate each configuration,
and therefore, this function can run on a distributed system transparently.

The `configurations` parameter can be an integer or a list of initial configurations (initial population); an integer
indicates the number of random individuales to be sampled from the search space.

The search space is defined with `config_kwargs` which correspond to those arguments of the `random_configurations` function.

The hyper-parameters found in this function are have the following meanings:
- `bsize`: the number of best items selected after each iteration to perform crossover (selection)
- `mutation_bsize`: the number of random items to be added to perform crossover (mutation)
- `ssize`: the size of the expected sample size from crossing individuals (offspring)
- `folds`: instructions for cross-validation, in particular we have the following possible kinds of inputs
  - ``0 < folds < 1`` a holdout partition (training with ``folds \\times n`` items and validating with ``(1-folds) \\times n`` items.
  - `folds` as a positive integer determines the `k` in `kfolds` crossvalidation
  - folds as an array of 2-tuples with indexes over `X` and `y` (manually given partitions)
- `search_maxiters`: sets the number of iterations before stop model selection.
- `score`: the score function to be optimized (a function or a symbol with a key over `scores`-function's output)
- `tol`: determines the tolerance; i.e., if the current iteration doesn't improves the previous one in at least `tol`
  regarding `score`, then the model selection procedure will be stopped. Set `tol` to a negative number to ignore this
  early stopping mode in favor of `search_maxiters`.
- `verbose`: indicates that you are willing to have a verbose output of several internal steps.
- `models`: if a dictionary is given, then all models and scores are captured into `models`.
- `distributed`: if it is true then the model evaluation is made with Distributed.@spawn (useful for debugging)
"""
function search_params(::Type{AKNC}, X, y, configurations;
        bsize::Integer=4,
        mutation_bsize::Integer=1,
        ssize::Integer=8,
        folds=0.7,
        search_maxiters::Integer=8,
        score=:macro_recall,
        tol::AbstractFloat=0.01,
        verbose=true,
        models::Union{Nothing,Dict}=nothing,
        distributed=false,
        config_kwargs...
    )
    
    save_models = models isa Dict
    if configurations isa Integer
       configurations = random_configurations(AKNC, nothing, configurations; config_kwargs...)
    end

    n = length(y)
    if folds isa Integer
        indexes = shuffle!(collect(1:n))
        folds = kfolds(indexes, folds)
    elseif folds isa AbstractFloat
        !(0.0 < folds < 1.0) && error("the folds parameter should follow 0.0 < folds < 1.0")
        indexes = shuffle!(collect(1:n))
        m = ceil(Int, folds * n)
        folds = [(indexes[1:m], indexes[m+1:end])]
    end
    
    if score isa Symbol
        scorefun = (perf) -> perf.scores[score]
    else
        scorefun = score::Function
    end
    
    prev = 0.0
    iter = 0
    while iter <= search_maxiters
        iter += 1
        C = AKNC_Config[]
        S = []

        for (config, score_) in configurations
            score_ >= 0.0 && continue
            push!(S, [])
            
            for (itrain, itest) in folds
                perf = if distributed
                    @spawn begin
                        p = evaluate_model(config, X[itrain], y[itrain], X[itest], y[itest], verbose=verbose)
                        save_models ? p : (scores=p.scores, model=nothing)
                    end
                else
                    p = evaluate_model(config, X[itrain], y[itrain], X[itest], y[itest], verbose=verbose)
                    save_models ? p : (scores=p.scores, model=nothing)
                end
                push!(S[end], perf)
            end
        
            push!(C, config)
        end
        
        verbose && println(stderr, "iteration $iter finished")

        for (c, perf_list) in zip(C, S)
            perf_list = fetch.(perf_list)
            if save_models
                models[c] = perf_list
            end
            configurations[c] = mean([scorefun(p) for p in perf_list])
        end

        if iter <= search_maxiters
            L = sort!(collect(configurations), by=x->x[2], rev=true)
            curr = L[1][2]
            if abs(curr - prev) <= tol                
                verbose && println(stderr, "stopping on iteration $iter due to a possible convergence ($curr ≃ $prev, tol: $tol)")
                break
            end

            prev = curr
            if verbose
                println(stderr, "generating $ssize configurations using top $bsize configurations, starting with $(length(configurations)))")
                println(stderr, [l[end] for l in L])
                println(stderr, L[1])
            end

            L =  AKNC_Config[L[i][1] for i in 1:min(bsize, length(L))]
            if mutation_bsize > 0
                for p in keys(random_configurations(AKNC, nothing, mutation_bsize; config_kwargs...))
                    push!(L, p)
                end
            end

            combine_configurations(L, ssize, configurations)
            verbose && println(stderr, "finished with $(length(configurations))")
        end
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end

