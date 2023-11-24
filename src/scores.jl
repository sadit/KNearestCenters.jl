# This file is a part of KNearestCenters.jl

using StatsBase
export accuracy_score, precision_recall, precision_score, recall_score, f1_score, classification_scores

## classification scores
"""
    recall_score(gold, predicted; weight=:macro)::Float64

It computes the recall between the gold dataset and the list of predictions `predict`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global recall, without distinguishing among classes
"""
function recall_score(gold, predicted; weight=:macro)::Float64
    P = precision_recall(gold, predicted)
    if weight === :macro
        mean(x -> x.recall, values(P.per_class))
    elseif weight === :weighted
        mean(x -> x.recall * x.population / length(gold), values(P.per_class))
    elseif :micro
        P.recall
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

"""
    precision_score(gold, predicted; weight=:macro)::Float64

It computes the precision between the gold dataset and the list of predictions `predict`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global precision, without distinguishing among classes
"""
function precision_score(gold, predicted; weight=:macro)::Float64
    P = precision_recall(gold, predicted)
    if weight === :macro
        mean(x -> x.precision, values(P.per_class))
    elseif weight === :weighted
        mean(x -> x.precision * x.population / length(gold), values(P.per_class))
    elseif weight === :micro
        P.precision
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

nan_zero(x) = isnan(x) ? zero(x) : x

function f1_(p, r)::Float64
    nan_zero(2 * p * r / (p + r))
end

"""
    f1_score(gold, predicted; weight=:macro)::Float64

It computes the F1 score between the gold dataset and the list of predictions `predicted`

It applies the desired weighting scheme for binary and multiclass problems
- `:macro` performs a uniform weigth to each class
- `:weigthed` the weight of each class is proportional to its population in gold
- `:micro` returns the global F1, without distinguishing among classes
"""
function f1_score(gold, predicted; weight=:macro)::Float64
    P = precision_recall(gold, predicted)
    if weight === :macro
        mean(x -> f1_(x.precision, x.recall), values(P.per_class))
    elseif weight === :weighted
        mean(x -> f1_(x.precision,x.recall) * x.population / length(gold), values(P.per_class))
    elseif weight === :micro
        f1_(P.precision, P.recall)
    else
        throw(Exception("Unknown weighting method $weight"))
    end
end

"""
    classification_scores(gold, predicted; labelnames=nothing)

Computes several scores for the given gold-standard and predictions, namely: 
precision, recall, and f1 scores, for global and per-class granularity.
If labelnames is given, then it is an array of label names.

"""
function classification_scores(gold::AbstractVector, predicted::AbstractVector; labelnames=nothing)
    class_f1 = Dict()
	  class_precision = Dict()
	  class_recall = Dict()

    P = precision_recall(gold, predicted)
    
    for (k, v) in P.per_class
        k = labelnames === nothing ? k : labelnames[k]
        class_f1[k] = f1_(v.precision, v.recall)
		class_precision[k] = v.precision
		class_recall[k] = v.recall
    end

    (
        microf1 = f1_(P.precision, P.recall),
    		precision = P.precision,
        macroprecision = mean(values(class_precision)),
        recall = P.recall,
        macrorecall = mean(values(class_recall)),
        macrof1 = mean(values(class_f1)),
        accuracy = accuracy_score(gold, predicted),
        classf1 = class_f1,
        classprecision = class_precision,
        classrecall = class_recall
    )
end

"""
    precision_recall(gold::AbstractVector, predicted::AbstractVector)

Computes the global and per-class precision and recall values between the gold standard
and the predicted set
"""
function precision_recall(gold::AbstractVector, predicted::AbstractVector)
    labels = unique(gold)
    M = Dict{typeof(labels[1]), NamedTuple}()
    tp_ = 0
    tn_ = 0
    fn_ = 0
    fp_ = 0

    for label in labels
        lgold = label .== gold
        lpred = label .== predicted

        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in 1:length(lgold)
            if lgold[i] == lpred[i]
                if lgold[i]
                    tp += 1
                else
                    tn += 1
                end
            else
                if lgold[i]
                    fn += 1
                else
                    fp += 1
                end
            end
        end

        tp_ += tp
        tn_ += tn
        fn_ += fn
        fp_ += fp
        precision = tp / (tp + fp)

        if isnan(precision)
            precision = 0.0
            @info "precision is zero for label '$label'; #classes=$(length(labels)) "
        end
        M[label] = (precision=precision, recall=tp / (tp + fn), population=sum(lgold) |> Int)
    end

    (precision=tp_ / (tp_ + fp_), recall=tp_ / (tp_ + fn_), per_class=M)
end

"""
    accuracy_score(gold, predicted)

Computes the accuracy score between the gold and the predicted sets
"""
function accuracy_score(gold::AbstractVector, predicted::AbstractVector)
    #  mean(gold .== predicted)
    c = 0
    for i in 1:length(gold)
        c += (gold[i] == predicted[i])
    end

    c / length(gold)
end

######### Regression ########

export pearson, spearman, isqerror
"""
    pearson(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}

Pearson correlation score
"""
function pearson(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}
    X̄ = mean(X)
    Ȳ = mean(Y)
    n = length(X)
    sumXY = 0.0
    sumX2 = 0.0
    sumY2 = 0.0
    for i in 1:n
        x, y = X[i], Y[i]
        sumXY += x * y
        sumX2 += x * x
        sumY2 += y * y
    end
    num = sumXY - n * X̄ * Ȳ
    den = sqrt(sumX2 - n * X̄^2) * sqrt(sumY2 - n * Ȳ^2)
    num / den
end

"""
    spearman(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}

Spearman rank correleation score
"""
function spearman(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}
    n = length(X)
    x = invperm(sortperm(X))
    y = invperm(sortperm(Y))
    d = x - y
    1 - 6 * sum(d.^2) / (n * (n^2 - 1))
end

"""
    isqerror(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}

Negative squared error (to be used for maximizing algorithms)
"""
function isqerror(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}
    n = length(X)
    d = 0.0

    @inbounds for i in 1:n
        d += (X[i] - Y[i])^2
    end

    -d
end

