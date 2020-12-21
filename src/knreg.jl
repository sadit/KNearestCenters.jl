# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt


mutable struct AKNR{ValType}
    nc::AKNC
    nclusters::Int
    label_value_map::Vector{ValType}
end

function fit(::Type{AKNR}, X, y, nclusters::Int; initial=:fft, dist::Function=l2_distance, centroid::Function=mean, summary::Function=mean, maxiters=10, tol=0.001, configurations=16, recall=1.0, verbose=false, kwargs...)
    C = kcenters(dist, X, nclusters, centroid; initial=initial, maxiters=maxiters, tol=tol, recall=recall, verbose=verbose)
    
    best_list = search_params(AKNC, X, C.codes, configurations;
        verbose=verbose,
        ncenters=[0],
        k=[1],
        dist=[dist],
        kwargs...
    )

    aknc = fit(best_list[1][1], X, C.codes; verbose=verbose)
    
end

