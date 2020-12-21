# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export size_criterion, sqrt_criterion, change_criterion, fun_criterion, log_criterion, epsilon_criterion, salesman_criterion


"""
    epsilon_criterion(e)

Creates a function that evaluates the stop criterion when the distance between far items achieves the given `e`
"""
function epsilon_criterion(e)
   (dmaxlist, database) -> dmaxlist[end] < e
end

"""
    size_criterion(maxsize)

Creates a function that stops when the number of far items are equal or larger than the given `maxsize`
"""
function size_criterion(maxsize)
   (dmaxlist, database) -> length(dmaxlist) >= maxsize
end

"""
    fun_criterion(fun::Function)

Creates a stop-criterion function that stops whenever the number of far items reaches ``\\lceil fun(|database|)\\rceil``.
Already defined examples:
```julia
    sqrt_criterion() = fun_criterion(sqrt)
    log2_criterion() = fun_criterion(log2)
```
"""
function fun_criterion(fun)
   (dmaxlist, database) -> length(dmaxlist) >= ceil(Int, length(database) |> fun |> round)
end

sqrt_criterion() = fun_criterion(sqrt)
log2_criterion() = fun_criterion(log2)

"""
    change_criterion(tol=0.001, window=3)

Creates a fuction that stops the process whenever the maximum distance converges (averaging `window` far items).
The `tol` parameter defines the tolerance range.
"""
function change_criterion(tol=0.001, window=3)
    mlist = Float64[]
    count = 0.0
    function stop(dmaxlist, database)
        count += dmaxlist[end]
        
        if length(dmaxlist) % window != 1
            return false
        end
        push!(mlist, count)
        count = 0.0
        if length(dmaxlist) < 2
            return false
        end
        
        s = abs(mlist[end] - mlist[end-1])
        return s <= tol
    end
    
    return stop
end

"""
    salesman_criterion()

It creates a function that explores the entire dataset making a full farthest first traversal approximation
"""
function salesman_criterion()
    function stop(dmaxlist, dataset)
        return false
    end
end