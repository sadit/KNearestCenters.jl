# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using KCenters
using MLDataUtils, Distributed, StatsBase
import StatsBase: fit, predict
import Base: hash, isequal
export search_params, random_configuration, combine_configurations, fit, after_load, predict, AKNC, AKNC_Config, AKNC_ConfigSpace

struct AKNC_Config{K_<:AbstractKernel, M_<:PreMetric}
    kernel::Type{K_}
    dist::Type{M_}
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

AKNC_Config(;
    kernel::Type=ReluKernel,
    dist::Type=CosineDistance,
    centroid::Function=mean,
    summary::Function=most_frequent_label,

    k::Int=1,
    ncenters::Integer=0,
    maxiters::Integer=1,
    
    recall::AbstractFloat=1.0,
    initial_clusters=:rand,
    split_entropy::AbstractFloat=0.6,
    minimum_elements_per_centroid=3
) = AKNC_Config(
        kernel, dist, centroid, summary, k, ncenters, maxiters,
        recall, initial_clusters, split_entropy, minimum_elements_per_centroid)

hash(a::AKNC_Config) = hash(repr(a))
isequal(a::AKNC_Config, b::AKNC_Config) = isequal(repr(a), repr(b))

struct AKNC{KNCType<:KNC, KernelType<:AbstractKernel}
    nc::KNCType
    kernel::KernelType
    config::AKNC_Config
end

"""
    struct AKNC{KNCType<:KNC, KernelType<:AbstractKernel}
        nc::KNCType
        kernel::KernelType
        config::AKNC_Config
    end

    AKNC(config::AKNC_Config, X, y; verbose=true)

Creates a new `AKNC` model using the given configuration and the dataset `X` and `y`
"""
function AKNC(config::AKNC_Config, X, y; verbose=true)
    kernel = config.kernel(config.dist())
    if config.ncenters == 0
        verbose && println("AKNC> clustering data with labels")
        C = kcenters(kernel.dist, X, y, config.centroid)
        knc = KNC(kernel, C)
    else
        verbose && println("AKNC> clustering data")
        C = kcenters(kernel.dist, X, config.ncenters, config.centroid,
            initial=config.initial_clusters, recall=config.recall, verbose=verbose, maxiters=config.maxiters)
        knc = KNC(kernel, C, X, y,
            config.centroid,
            split_entropy=config.split_entropy,
            minimum_elements_per_centroid=config.minimum_elements_per_centroid,
            verbose=verbose)
    end

    AKNC(knc, kernel, config)
end

"""
    predict(model::AKNC, x, res::KnnResult=model.nc.res)

Predicts the label of each item in `X` using `model`; k == 0 means for using the stored `k` in `config`
"""

function predict(model::AKNC, x, res::KnnResult=model.nc.res)
    empty!(res, model.config.k)
    predict(model.nc, x, res; summary=model.config.summary)
end

Base.broadcastable(model::AKNC) = (model,)

"""
    evaluate_model(config::AKNC_Config, train_X, train_y::CategoricalArray, test_X, test_y::CategoricalArray; verbose=true)

Creates a model for `train_X` and `train_y`, defined with `config`, evaluates them with `test_X` and `test_y`.
Returns a named tuple containing the evalution scores and the computed model.
"""
function evaluate_model(config::AKNC_Config, train_X, train_y::CategoricalArray, test_X, test_y::CategoricalArray; verbose=true)
    knc = AKNC(config, train_X, train_y, verbose=verbose)
    ypred = [predict(knc, x) for x in test_X]
    s = classification_scores(test_y.refs, ypred)
    if verbose
        println(stderr, typeof(test_y), typeof(ypred))
        println(stderr, s)
    end
    (scores=s, model=knc)
end

struct AKNC_ConfigSpace
    kernel::Vector{Type}
    dist::Vector{Type}
    centroid::Vector{Function}
    summary::Vector{Function}
    k::Vector{Integer}
    maxiters::Vector{Integer}
    recall::Vector{Real}
    ncenters::Vector{Integer}
    initial_clusters::Vector{Any}
    split_entropy::Vector{Real}
    minimum_elements_per_centroid::Vector{Integer}
end

"""
    AKNC_ConfigSpace(;
        kernel::AbstractVector=[RelyKernel, DirectKernel], 
        dist::AbstractVector=[L2Distance, CosineDistance],
        centroid::AbstractVector=[mean],
        summary::AbstractVector=[most_frequent_label, mean_label],
        k::AbstractVector=[1],
        maxiters::AbstractVector=[1, 3, 10],
        recall::AbstractVector=[1.0],
        ncenters::AbstractVector=[0, 10],
        initial_clusters::AbstractVector=[:fft, :dnet, :rand],
        split_entropy::AbstractVector=[0.3, 0.6, 0.9],
        minimum_elements_per_centroid::AbstractVector=[1, 3, 5]
    )

Creates a configuration space for AKNC_Config
"""
AKNC_ConfigSpace(;
    kernel::Vector=[ReluKernel, DirectKernel, GaussianKernel],
    dist::Vector=[L2Distance, CosineDistance],
    centroid::Vector=[mean],
    summary::Vector=[most_frequent_label, mean_label],
    k::Vector=[1],
    maxiters::Vector=[1, 3, 10],
    recall::Vector=[1.0],
    ncenters::Vector=[0, 10],
    initial_clusters::Vector=[:fft, :dnet, :rand],
    split_entropy::Vector=[0.3, 0.6, 0.9],
    minimum_elements_per_centroid::Vector=[1, 3, 5]
) = AKNC_ConfigSpace(kernel, dist, centroid, summary, k, maxiters, recall, ncenters, initial_clusters, split_entropy, minimum_elements_per_centroid)

"""
    random_configuration(space::AKNC_ConfigSpace)

Creates a random `AKNC_Config` instance based on the `space` definition.
"""
function random_configuration(space::AKNC_ConfigSpace)
    ncenters = rand(space.ncenters)

    if ncenters == 0
        maxiters = 0
        split_entropy = 0.0
        minimum_elements_per_centroid = 1
        initial_clusters = :rand  # nothing in fact
        k = 1
    else
        maxiters = rand(space.maxiters)
        split_entropy = rand(space.split_entropy)
        minimum_elements_per_centroid = rand(space.minimum_elements_per_centroid)
        initial_clusters = rand(space.initial_clusters)
        k = rand(space.k)
    end

    config = AKNC_Config(
        kernel = rand(space.kernel),
        dist = rand(space.dist),
        centroid = rand(space.centroid),
        summary = rand(space.summary),
        k = k,
        ncenters = ncenters,
        maxiters = maxiters,
        recall = rand(space.recall),
        initial_clusters = initial_clusters,
        split_entropy = split_entropy,
        minimum_elements_per_centroid = minimum_elements_per_centroid
    )
end

"""
    combine_configurations(config_list::AbstractVector{AKNC_Config})

Creates a new configuration combining the given configurations
"""
function combine_configurations(config_list::AbstractVector{AKNC_Config})
    _sel() = rand(config_list)

    a = _sel()  # select a basis element
    AKNC_Config(
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
end

"""
    function search_params(config_space::AKNC_ConfigSpace, X, y, m=8;
        configurations=Dict{AKNC_Config,Float64}(),
        bsize::Integer=4,
        mutation_bsize::Integer=1,
        ssize::Integer=8,
        folds=0.7,
        search_maxiters::Integer=8,
        score=:macro_recall,
        tol::AbstractFloat=0.01,
        verbose=true,
        modelstorage::Union{Nothing,Dict}=nothing,
        distributed=false
    ) config_kwargs...
    )

Performs a model selection of AKNC for the given examples ``(X, y)`` using an evolutive algorithm
(a variant of a genetic algorithm). Note that this function use Distributed's `@spawn` to evaluate each configuration,
and therefore, this function can run on a distributed system transparently.

The `m` integer indicates the number of random individuales to be sampled from the search space.
The `configurations` may be given as an initial set of configurations.


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
- `modelstorage`: if a dictionary is given, then all models and scores are captured into.
- `distributed`: if it is true then the model evaluation is made with Distributed.@spawn (useful for debugging)
"""
function search_params(config_space::AKNC_ConfigSpace, X, y, m=8;
        configurations=Dict{AKNC_Config,Float64}(),
        bsize::Integer=4,
        mutation_bsize::Integer=1,
        ssize::Integer=8,
        folds=0.7,
        search_maxiters::Integer=8,
        score=:macro_recall,
        tol::AbstractFloat=0.01,
        verbose=true,
        modelstorage::Union{Nothing,Dict}=nothing,
        distributed=false
    )
    
    save_models = modelstorage !== nothing

    # initializing population
    for i in 1:m
        configurations[random_configuration(config_space)] = -1.0
    end

    n = length(y)
    if folds isa Integer
        indexes = Random.shuffle!(collect(1:n))
        folds = kfolds(indexes, folds)
    elseif folds isa AbstractFloat
        !(0.0 < folds < 1.0) && error("the folds parameter should follow 0.0 < folds < 1.0")
        indexes = Random.shuffle!(collect(1:n))
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
                modelstorage[c] = perf_list
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

            # preparing for crossover best items and some random configurations
            L =  AKNC_Config[L[i][1] for i in 1:min(bsize, length(L))] # select best
            for i in 1:mutation_bsize
                push!(L, random_configuration(config_space))
            end
        
            for i in 1:ssize
                conf = combine_configurations(L)
                if !haskey(configurations, conf)
                    configurations[conf] = -1.0
                end
            end
            verbose && println(stderr, "finished with $(length(configurations))")
        end
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end

