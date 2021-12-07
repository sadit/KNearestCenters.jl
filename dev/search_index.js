var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = KNearestCenters","category":"page"},{"location":"#KNearestCenters","page":"Home","title":"KNearestCenters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The KNearestCenters package contains classification algorithms based on prototype selection and feature mapping through kernel functions. It model selection to improve the classification performance.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [KNearestCenters]","category":"page"},{"location":"#KNearestCenters.Knc","page":"Home","title":"KNearestCenters.Knc","text":"A nearest centroid classifier with support for kernel functions\n\n\n\n\n\n","category":"type"},{"location":"#KNearestCenters.Knc-Tuple{KncConfig, Any, CategoricalArrays.CategoricalArray}","page":"Home","title":"KNearestCenters.Knc","text":"Knc(config::KncConfig, X, y::CategoricalArray; verbose=true)\n\nCreates a Knc classifier using the given configuration and data.\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.KncProto","page":"Home","title":"KNearestCenters.KncProto","text":"A simple nearest centroid classifier with support for kernel functions\n\n\n\n\n\n","category":"type"},{"location":"#KNearestCenters.KncProto-Tuple{KncProtoConfig, Any, CategoricalArrays.CategoricalArray}","page":"Home","title":"KNearestCenters.KncProto","text":"KncProto(config::KncProtoConfig, X, y::CategoricalArray; verbose=true)\nKncProto(config::KncProtoConfig,\n    input_clusters::ClusteringData,\n    train_X::AbstractVector,\n    train_y::CategoricalArray;\n    verbose=false\n)\n\nCreates a KncProto classifier using the given configuration and data.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{KncConfigSpace}","page":"Home","title":"Base.rand","text":"rand(space::KncConfigSpace)\n\nCreates a random KncConfig instance based on the space definition.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Union{Tuple{KncProtoConfigSpace{KindProto, MutProb}}, Tuple{MutProb}, Tuple{KindProto}} where {KindProto, MutProb}","page":"Home","title":"Base.rand","text":"rand(space::KncProtoConfigSpace)\n\nCreates a random KncProtoConfig instance based on the space definition.\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{CauchyKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::CauchyKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Cauchy kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{DirectKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::DirectKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Direct kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{GaussianKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::GaussianKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Gaussian kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{LaplacianKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::LaplacianKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Laplacian kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{ReluKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::ReluKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Relu kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{SigmoidKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::SigmoidKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Sigmoid kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{TanhKernel, Any, Any, AbstractFloat}","page":"Home","title":"Distances.evaluate","text":"evaluate(kernel::TanhKernel, a, b, σ::AbstractFloat)::Float64\n\nCreates a Tanh kernel with the given distance function\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.accuracy_score-Tuple{AbstractVector, AbstractVector}","page":"Home","title":"KNearestCenters.accuracy_score","text":"accuracy_score(gold, predicted)\n\nComputes the accuracy score between the gold and the predicted sets\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.change_criterion","page":"Home","title":"KNearestCenters.change_criterion","text":"change_criterion(tol=0.001, window=3)\n\nCreates a fuction that stops the process whenever the maximum distance converges (averaging window far items). The tol parameter defines the tolerance range.\n\n\n\n\n\n","category":"function"},{"location":"#KNearestCenters.classification_scores-Tuple{AbstractVector, AbstractVector}","page":"Home","title":"KNearestCenters.classification_scores","text":"classification_scores(gold, predicted; labelnames=nothing)\n\nComputes several scores for the given gold-standard and predictions, namely:  precision, recall, and f1 scores, for global and per-class granularity. If labelnames is given, then it is an array of label names.\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.epsilon_criterion-Tuple{Any}","page":"Home","title":"KNearestCenters.epsilon_criterion","text":"epsilon_criterion(e)\n\nCreates a function that evaluates the stop criterion when the distance between far items achieves the given e\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.f1_score-Tuple{Any, Any}","page":"Home","title":"KNearestCenters.f1_score","text":"f1_score(gold, predicted; weight=:macro)::Float64\n\nIt computes the F1 score between the gold dataset and the list of predictions predicted\n\nIt applies the desired weighting scheme for binary and multiclass problems\n\n:macro performs a uniform weigth to each class\n:weigthed the weight of each class is proportional to its population in gold\n:micro returns the global F1, without distinguishing among classes\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.fun_criterion-Tuple{Any}","page":"Home","title":"KNearestCenters.fun_criterion","text":"fun_criterion(fun::Function)\n\nCreates a stop-criterion function that stops whenever the number of far items reaches lceil fun(database)rceil. Already defined examples:\n\n    sqrt_criterion() = fun_criterion(sqrt)\n    log2_criterion() = fun_criterion(log2)\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.isqerror-Union{Tuple{F}, Tuple{AbstractVector{F}, AbstractVector{F}}} where F<:AbstractFloat","page":"Home","title":"KNearestCenters.isqerror","text":"isqerror(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}\n\nNegative squared error (to be used for maximizing algorithms)\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.mean_label-Tuple{KncProto, SimilaritySearch.KnnResult}","page":"Home","title":"KNearestCenters.mean_label","text":"mean_label(nc::KncProto, res::KnnResult)\n\nSummary function that computes the label as the mean of the k nearest labels (ordinal classification)\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.most_frequent_label-Tuple{KncProto, SimilaritySearch.KnnResult}","page":"Home","title":"KNearestCenters.most_frequent_label","text":"most_frequent_label(nc::KncProto, res::KnnResult)\n\nSummary function that computes the label as the most frequent label among labels of the k nearest prototypes (categorical labels)\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.pearson-Union{Tuple{F}, Tuple{AbstractVector{F}, AbstractVector{F}}} where F<:AbstractFloat","page":"Home","title":"KNearestCenters.pearson","text":"pearson(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}\n\nPearson correlation score\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.precision_recall-Tuple{AbstractVector, AbstractVector}","page":"Home","title":"KNearestCenters.precision_recall","text":"precision_recall(gold::AbstractVector, predicted::AbstractVector) where {T1<:Integer} where {T2<:Integer\n\nComputes the global and per-class precision and recall values between the gold standard and the predicted set\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.precision_score-Tuple{Any, Any}","page":"Home","title":"KNearestCenters.precision_score","text":"precision_score(gold, predicted; weight=:macro)::Float64\n\nIt computes the precision between the gold dataset and the list of predictions predict\n\nIt applies the desired weighting scheme for binary and multiclass problems\n\n:macro performs a uniform weigth to each class\n:weigthed the weight of each class is proportional to its population in gold\n:micro returns the global precision, without distinguishing among classes\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.recall_score-Tuple{Any, Any}","page":"Home","title":"KNearestCenters.recall_score","text":"recall_score(gold, predicted; weight=:macro)::Float64\n\nIt computes the recall between the gold dataset and the list of predictions predict\n\nIt applies the desired weighting scheme for binary and multiclass problems\n\n:macro performs a uniform weigth to each class\n:weigthed the weight of each class is proportional to its population in gold\n:micro returns the global recall, without distinguishing among classes\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.salesman_criterion-Tuple{}","page":"Home","title":"KNearestCenters.salesman_criterion","text":"salesman_criterion()\n\nIt creates a function that explores the entire dataset making a full farthest first traversal approximation\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.size_criterion-Tuple{Any}","page":"Home","title":"KNearestCenters.size_criterion","text":"size_criterion(maxsize)\n\nCreates a function that stops when the number of far items are equal or larger than the given maxsize\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.softmax!-Tuple{AbstractVector}","page":"Home","title":"KNearestCenters.softmax!","text":"softmax!(vec::AbstractVector)\n\nInline computation of the softmax function on the input vector\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.spearman-Union{Tuple{F}, Tuple{AbstractVector{F}, AbstractVector{F}}} where F<:AbstractFloat","page":"Home","title":"KNearestCenters.spearman","text":"spearman(X::AbstractVector{F}, Y::AbstractVector{F}) where {F <: AbstractFloat}\n\nSpearman rank correleation score\n\n\n\n\n\n","category":"method"},{"location":"#KNearestCenters.transform","page":"Home","title":"KNearestCenters.transform","text":"transform(nc::Knc, kernel::Function, X, normalize!::Function=softmax!)\n\nMaps a collection of objects to the vector space defined by each center in nc; the kernel function is used measure the similarity between each u in X and each center in nc. The normalization function is applied to each vector (normalization methods needing to know the attribute's distribution can be applied on the output of transform)\n\n\n\n\n\n","category":"function"},{"location":"#SearchModels.combine-Tuple{KncConfig, KncConfig}","page":"Home","title":"SearchModels.combine","text":"combine(a::KncConfig, b::KncConfig)\n\nCreates a new configuration combining the given configurations\n\n\n\n\n\n","category":"method"},{"location":"#SearchModels.combine-Tuple{KncProtoConfig, KncProtoConfig}","page":"Home","title":"SearchModels.combine","text":"combine(a::KncProtoConfig, b::KncProtoConfig)\n\nCreates a new configuration combining the given configurations\n\n\n\n\n\n","category":"method"},{"location":"#SearchModels.mutate-Tuple{KncProtoConfigSpace, KncProtoConfig, Any}","page":"Home","title":"SearchModels.mutate","text":"mutate(space::KncProtoConfigSpace, a::KncProtoConfig, iter)\n\nCreates a new configuration based on a slight perturbation of a\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.predict-Tuple{KncProto, Any, SimilaritySearch.KnnResult}","page":"Home","title":"StatsBase.predict","text":"predict(nc::KncProto, x, res::KnnResult)\npredict(nc::KncProto, x)\n\nPredicts the class of x using the label of the k nearest centers under the kernel function.\n\n\n\n\n\n","category":"method"}]
}
