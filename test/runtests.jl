# This file is a part of KCenters.jl

using Test
using KNearestCenters

@testset "Scores" begin
    @test accuracy_score([1,1,1,1,1], [1,1,1,1,1]) == 1.0
    @test accuracy_score([1,1,1,1,1], [0,0,0,0,0]) == 0.0
    @test accuracy_score([1,1,1,1,0], [0,1,1,1,1]) == 0.6
    @test precision_recall([0,1,1,1,0,1], [0,1,1,1,1,1]) == (precision=0.8333333333333334, recall=0.8333333333333334,
        per_class=Dict(0 => (precision=1.0, recall=0.5, population=2), 1 => (precision=0.8, recall=1.0, population=4)))
    @test precision_score([0,1,1,1,0,1], [0,1,1,1,1,1]) == 0.9
    @test recall_score([0,1,1,1,0,1], [0,1,1,1,1,1]) == 0.75
    @test precision_score([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:weighted) == (1.0 * 2/6 + 0.8 * 4/6) / 2
    @test recall_score([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:weighted) == (0.5 * 2/6 + 1.0 * 4/6) / 2
    @test f1_score([0,1,1,1,0,1], [0,1,1,1,1,1], weight=:macro) ≈ (2 * 0.5 / 1.5 + 2 * 0.8 / 1.8) / 2
end

include("knc.jl")
include("kncproto.jl")
