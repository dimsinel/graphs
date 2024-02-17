using DrWatson
@quickactivate "graphs"


# Load the module and generate the functions
module CppHello
using CxxWrap
@wrapmodule(() -> joinpath("/home/lenny/Work/libcxxwrap-julia/build/lib", "libhello"))

function __init__()
    @initcxx
end
end


module Basic_Types
using CxxWrap
@wrapmodule(() -> joinpath("/home/lenny/Work/libcxxwrap-julia/build/lib", "libbasic_types"))

function __init__()
    @initcxx
end
end
# Call greet and show the result

@show CppHello.greet()

CppHello.add(1.1, 2)


using MLUtils

# X is a matrix of floats
# Y is a vector of strings
X, Y = load_iris()

# The iris dataset is ordered according to their labels,
# which means that we should shuffle the dataset before
# partitioning it into training- and test-set.
Xs, Ys = shuffleobs((X, Y))

# We leave out 15 % of the data for testing
cv_data, test_data = splitobs((Xs, Ys); at=0.85)

# Next we partition the data using a 10-fold scheme.
for (train_data, val_data) in kfolds(cv_data; k=10)

    # We apply a lazy transform for data augmentation
    train_data = mapobs(xy -> (xy[1] .+ 0.1 .* randn.(), xy[2]), train_data)

    for epoch = 1:10
        # Iterate over the data using mini-batches of 5 observations each
        for (x, y) in eachobs(train_data, batchsize=5)
            @show x, y
            # ... train supervised model on minibatches here
        end
    end
end

using MLJ
measures("FScore")
m1 = FScore() #levels=[0,1])
m2 = MulticlassFScore()

y₀ = coerce(categorical(rand("abc", 10)), Multiclass)
#y₀ = categorical(rand("abc", 10))
y = coerce(categorical(rand("abc", 10)), Multiclass) # or `coerce(rand("abc", 10), Multiclass))`
#ŷ = categorical(rand("abc", 10))
ŷ = coerce(categorical(rand("abc", 10)), Multiclass)  # or `coerce(rand("abc", 10), Multiclass))`

MulticlassFScore()(ŷ, y)
#MulticlassFScore()(ŷ, y, class_w)



m1(y, ŷ)
m1(y₀, ŷ)

m2(y, ŷ)
m2(y₀, ŷ)

cm = ConfusionMatrix()(ŷ, y)  # or `confmat((ŷ, y)`.

## scikit-learn example 
## at 
## http://tinyurl.com/yrc2vxds
#= 
import numpy as np
from sklearn.metrics import f1_score =#
y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 0, 1]
levels(y_true)
#=
f1_score(y_true, y_pred, average='macro')  ---> 0.26666666666666666
f1_score(y_true, y_pred, average='micro') ---> 0.3333333333333333
f1_score(y_true, y_pred, average='weighted')---> 0.26666666666666666
f1_score(y_true, y_pred, average=None)---> array([0.8, 0. , 0. ])
=#
m1 = FScore(levels=[0, 1])
m1(y_true, y_pred)
m2(y_true, y_pred)
# but MulticlassFscore has 3 aliases.
micro_f1score(y_true, y_pred)
macro_f1score(y_true, y_pred)
multiclass_f1score(y_true, y_pred)
confmat(y_true, y_pred)


using PyCall
using StatsPlots

py"""
import numpy as np
def compute_f1_score(hyperedge, predicted_hyperedge):
    common_part = np.bitwise_and(hyperedge, predicted_hyperedge)
    #print('--- >  ', common_part, hyperedge, predicted_hyperedge)
    f1_score = 0

    true_positives = sum(common_part)

    if true_positives > 0:
        #true_positives = len(common_part)
        total_positives = sum(predicted_hyperedge)
        actual_positives = sum(hyperedge)

        precision = true_positives / total_positives
        recall = true_positives / actual_positives

        f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score
"""

# symmetric
py"compute_f1_score"(y_true, y_pred)
py"compute_f1_score"(y_pred, y_true)
# this does not agree w/ any of the above


### Another try 

microf1 = Vector{Float64}(undef, 0)
macrof1 = Vector{Float64}(undef, 0)
pyf1 = Vector{Float64}(undef, 0)
pyf1R = Vector{Float64}(undef, 0)
m1f1 = Vector{Float64}(undef, 0)
m1f1R = Vector{Float64}(undef, 0)

for i in 1:10000
    y_true = rand([0, 1], 30)
    y_pred = rand([0, 1], 30)

    # confmat(y_true, y_pred)
    pyf = py"compute_f1_score"(y_true, y_pred)
    pyfR = py"compute_f1_score"(y_pred, y_true)
    jmacrof = macro_f1score(y_true, y_pred)
    #jmicrof = m1(y_true, y_pred)
    jmicrof = micro_f1score(y_true, y_pred)
    m1f = m1(y_true, y_pred)
    m1fR = m1(y_pred, y_true)
    #println("py $(pyf),  macro = $(jmacrof),  micro = $(jmicrof)")
    # m1(y_true, y_pred)
    # m2(y_true, y_pred)

    # but MulticlassFscore has 3 aliases.
    #micro_f1score(y_true, y_pred)
    push!(microf1, jmicrof)
    push!(macrof1, jmacrof)
    #multiclass_f1score(y_true, y_pred)
    #confmat(y_true, y_pred)
    push!(pyf1, pyf)
    push!(pyf1R, pyfR)
    push!(m1f1, m1f)
    push!(m1f1R, m1fR)
end

histogram(macrof1, label="macro") #, bins = .2:.01:.3)

#############################
histogram(pyf1, label="python") #, bins = .2:.01:.3)
histogram!(m1f1, label="m1") #, bins = .2:.01:.3)
############################
histogram!(m1f1R, label="m1R") #, bins = .2:.01:.3)
# m1 symmetric
sum(m1f1R - m1f1)
# pyf symmetric
sum(pyf1R - pyf1)

h_micro = histogram!(microf1, label="micro")

import StatsBase, Distributions
length(pyf1), typeof(pyf1)
h_microfit = StatsBase.fit(Distributions.Normal, pyf1)

# this must be a pdf or something, look it up
plot!(h_microfit, linewidth=4)

a = [1 2 3; 4 5 6]
a[1, 2], a[2, 1]


clf = ConstantClassifier()
X, y = @load_crabs # a table and a categorical vector
mach = machine(clf, X, y) |> fit!

fitted_params(mach)
Xnew = (; FL=[8.1, 24.8, 7.2],
    RW=[5.1, 25.7, 6.4],
    CL=[15.9, 46.7, 14.3],
    CW=[18.7, 59.7, 12.2],
    BD=[6.2, 23.6, 8.4],)

yhat = predict(mach, Xnew)
yhat[1]

# raw probabilities:
pdf.(yhat, "B")

# probability matrix:
L = levels(y)
pdf(yhat, L)

# point predictions:
predict_mode(mach, Xnew)




bit(x, y) = @pipe (x & x) |> (digits(_, base=2, pad=8) |> reverse)'


using Tullio, Test
M = rand(1:20, 3, 7)

@tullio S[1, c] := M[r, c]  # sum over r ∈ 1:3, for each c ∈ 1:7
@test S == sum(M, dims=1)

@tullio Q[ρ, c] := M[ρ, c] + sqrt(S[1, c])  # loop over ρ & c, no sum -- broadcasting
@test Q ≈ M .+ sqrt.(S)

mult(M, Q) = @tullio P[x, y] := M[x, c] * Q[y, c]  # sum over c ∈ 1:7 -- matrix multiplication
@test mult(M, Q) ≈ M * transpose(Q)

R = [rand(Int8, 3, 4) for δ in 1:5]

@tullio T[j, i, δ] := R[δ][i, j] + 10im  # three nested loops -- concatenation
@test T == permutedims(cat(R...; dims=3), (2, 1, 3)) .+ 10im

@tullio (max) X[i] := abs2(T[j, i, δ])  # reduce using max, over j and δ
@test X == dropdims(maximum(abs2, T, dims=(1, 3)), dims=(1, 3))

dbl!(M, S) = @tullio M[r, c] = 2 * S[1, c]  # write into existing matrix, M .= 2 .* S
dbl!(M, S)
@test all(M[r, c] == 2 * S[1, c] for r ∈ 1:3, c ∈ 1:7)

using BenchmarkTools, Test
using Pipe: @pipe
using LinearAlgebra, SparseArrays

mult(M, Q) = @tullio P[x, y] := M[x, c] * Q[y, c]  # sum over c ∈ 1:7 -- matrix multiplication


foreach((30, 50, 200)) do N
    A, B, C = randn(N, N + 1), randn(N + 1, N + 2), randn(N + 2, N + 3)
    CC = similar(C)
    CC = transpose(C)
    D = Matrix{Float64}(undef, N, N + 2)
    E = Matrix{Float64}(undef, N, N + 3)
    @show N
    #o1 = @btime triple_mul!($D, $A, $B, $C)
    @btime $(mul!(E, mul!(D, A, B), C))
    #o2 = @btime $A * $B * $C
    o2 = @btime $A * $B * $(transpose(CC))
    @test E == o2
    #@info "$(size(o1)), C-> $(size(C)) CC->$(size(CC)), $(size(mult(A, B')))"
    #aa = mult(A, B')
    #@info "$(size(aa))"
    o3 = @btime $(@pipe mult(A, B') |> mult(_, C'))
    #@info "$(size(o3))"
    #display(o1)
    #display(o3)
    @test o2 ≈ o3
    @info "sum(o1-03) = $(sum(E-o3)) "
end
#

begin
    n = 100
    a1 = sparse(Matrix(1.0I, n, n))
    a2 = Diagonal(ones(n))
    a3 = spzeros(n, n)
    a4 = sprand(n, n, 0.25)
    a41 = sprand(n, n, 0.25)
    a5 = rand(n, n)
    a6 = rand(n, n)

    @info "sparse diag × diag"
    @btime a1 * a2
    @info "sparse rand × diag"
    @btime a4 * a2
    @info "sparse diag × rand"
    @btime a1 * a6

    @info "sparse rand × rand"
    @btime a4 * a5
    @info "rand × rand"
    @btime a5 * a6
    @info "sparse rand × sparse rand"
    @btime a4 * a41

    @info "mul!"
    @btime mul!(a4, mul!(a1, a4, a6), a41)
end

########################################################################

using DrWatson
@quickactivate "graphs"
using LinearAlgebra, SparseArrays, StaticArrays

struct Arr{T}
    a::AbstractMatrix{T}
end
struct iArr
    a::AbstractMatrix{Int64}
end
aa = [1.0 2.0; 3 4]
sa = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])
sta = SMatrix{size(aa)...}([1 2; 3 4])

function Arr(matr::AbstractMatrix{T}) where {T}
    m = nothing
    if length(matr) < 65
        m = SMatrix{size(matr)...}(matr)
    elseif length(matr) < 100_000
        m = matr
    else
        m = sparse(matr)
    end
    Arr{T}(m)
end


function makearr(n, m, d)
    @assert d <= 1.0
    on = floor(n * m * d) |> Int64
    ze = n * m - on
    coll = vcat(zeros(ze), rand(on))
    #@show ze, on, length(coll)
    c = rand(coll, n, m)
end

a = makearr(200, 500, 0.1);
aa = Arr(a)
aa.a


using Tullio, BenchmarkTools, LoopVectorization
# Row major
a = Array(reshape(Int32.(1:2*2000*400), 2, 2000, 400));
b = Array(reshape(Int32.(1:2*2000*400), 2, 400, 2000));
@btime @tullio c[i, j, k] := $a[i, j, q] * $b[i, q, k]; # 682ms


using Tullio, BenchmarkTools, LoopVectorization
# Column major
a = Array(reshape(Int32.(1:2*2000*400), 2000, 400, 2));
b = Array(reshape(Int32.(1:2*2000*400), 400, 2000, 2));
@btime @tullio c[j, k, i] := $a[j, q, i] * $b[q, k, i]; # 126ms

using LinearAlgebra, SparseArrays, StaticArrays

abstract type AA{T, U} end


struct Aa{T,U} 
    a::T
    b::U

end
a1 = Aa{Matrix,Matrix}
a1.a
