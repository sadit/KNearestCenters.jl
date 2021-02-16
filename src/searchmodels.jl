# This file is a part of TextClassification.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export AbstractConfigSpace, AbstractConfig, search_models, random_configuration, combine_configurations, evaluate_model
using Distributed, Random, StatsBase
abstract type AbstractConfigSpace end
abstract type AbstractConfig end

Base.hash(a::AbstractConfig) = hash(repr(a))
Base.isequal(a::AbstractConfig, b::AbstractConfig) = isequal(repr(a), repr(b))
#function Base.eltype(space) => kind of configuration

#function random_configuration(space::AbstractConfigSpace) end
#function combine_configurations(space::AbstractConfigSpace, config_list::AbstractVector) end

function random_configuration end
function combine_configurations end

function search_models(
        configspace::AbstractConfigSpace,
        evaluate_model::Function,  # receives only the configuration to be evaluated
        m=8;
        configurations=Dict{AbstractConfig,Float64}(),
        bsize=16,
        mutbsize=4,
        crossbsize=4,
        maxiters=8,
        tol=0.01,
        verbose=true,
        distributed=false
    )
    
    for i in 1:m
        configurations[random_configuration(configspace)] = -1.0
    end
    
    prev = 0.0
    iter = 0
    while iter <= maxiters
        iter += 1

        verbose && println(stderr, "SearchModels> ==== search params iter=$iter, tol=$tol, m=$m, bsize=$bsize, mutbsize=$mutbsize, crossbsize=$crossbsize, prev=$prev, $(length(configurations))")
        S = Pair[]
        for (config, score_) in configurations
            score_ >= 0.0 && continue
            perf = if distributed
                @spawn evaluate_model(config)
            else
                evaluate_model(config)
            end
            push!(S, config => perf)
        end
        
        # fetching all results
        for s in S
            configurations[s.first] = fetch(s.second)
        end

        verbose && println(stderr, "SearchModels> *** iteration $iter finished; starting combinations.")

        iter > maxiters && break

        L = sort!(collect(configurations), by=x->x[2], rev=true)
        curr = L[1][2]
        if abs(curr - prev) <= tol     
            verbose && println(stderr, "SearchModels> *** stopping on iteration $iter due to a possible convergence ($curr ≃ $prev, tol: $tol)")
            break
        end

        prev = curr
        if verbose
            println(stderr, "SearchModels> *** adding more items to the population: bsize=$bsize; #configurations=$(length(configurations)))")
            println(stderr, "SearchModels> *** scores: ", [l[end] for l in L])
            config__, score__ = L[1]
            println(stderr, "SearchModels> *** best config with score $score__: ", [(k => getfield(config__, k)) for k in fieldnames(typeof(config__))])
        end

        L =  AbstractConfig[L[i][1] for i in 1:min(bsize, length(L))]
    
        for i in 1:mutbsize
            conf = combine_configurations(rand(L), random_configuration(configspace))
            if !haskey(configurations, conf)
                configurations[conf] = -1.0
            end
        end

        for i in 1:crossbsize
            conf = combine_configurations(rand(L), rand(L))
            if !haskey(configurations, conf)
                configurations[conf] = -1.0
            end
        end

        verbose && println(stderr, "SearchModels> *** finished with $(length(configurations)) configurations")
    
    end

    sort!(collect(configurations), by=x->x[2], rev=true)
end
