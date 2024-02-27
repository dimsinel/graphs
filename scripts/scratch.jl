using DrWatson
@quickactivate "graphs"
###############################################


#hyperedges = Vector{HyperEdge}(undef, n)
hyperedges = Vector{HyperEdge}()

new_hedges_sizes(i) = sample(diag(hg.h_edges), i)

function first_nodes(i)
    nodes_distribution = (hg.nodes |> diag |> fweights)
    return sample(1:hg.v_id, nodes_distribution, i)
end

#while length(hyperedges) < n
# choose the degrees of the new hyperedges according to current h_edge degrees
new_hedge_size = new_hedges_sizes(1)
#@show new_hedge_size
# choose new first nodes according to current nodes degree  using Preferential Attachment
# we dont care about the current no of nodes, only their degrees, (should we use 'unique' here?)
first_node = first_nodes(1)[1]
#@show any(==(0), first_nodes) 
#sanity
#do this for n > 1000 to chech if node sampling was OK: d should be approx nodes_distribution
#= begin) # |> unique)
# we will sample according to this distribution, 
nodes_distribution = (nodes_distribution |> fw
    ufn = unique(first_nodes)
    fnode_distr = map(i -> count(==(i), first_nodes), ufn)
    dd = Dict(zip(ufn, fnode_distr)) |> sort
    norm = dd[1]
    d = Vector{Float64}(undef, length(dd))
    for (i, k) in dd
        d[i] = dd[i] / norm * nodes_distribution[1]
    end
    @show unique(first_nodes), fnode_distr, nodes_distribution
    @show d
end =#

# Create a new h-edge
# well, all of them are going to have the same hyperedge number, but it is immaterial here.
new_he = HyperEdge(hg.e_id + 1, hg.v_id, Dict([first_node => 1]), 1.0)

# Now we must add to each hyperedge, new_hedge_size-1 nodes.
# a loop over the needed number of elements in the new h-edge 
new_Vertex = 0
while length(new_he.nodes) < new_hedge_size[1]

    new_Vertex = choose_new_vertex(hg, new_he)
    if new_Vertex == 0
        # something went wrong in the new vertex calculation. 
        # Start again 
        break
    end
    new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what
    keys(new_he.nodes)
end


# add the new vertex
new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what

# double check: 
if new_he ∉ hyperedges
    push!(hyperedges, new_he)

end

#end
#@show length(new_hedge_size), length(hyperedges)


###########################################################3




############################################################3
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

for i = 1:10000
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
Xnew = (;
    FL=[8.1, 24.8, 7.2],
    RW=[5.1, 25.7, 6.4],
    CL=[15.9, 46.7, 14.3],
    CW=[18.7, 59.7, 12.2],
    BD=[6.2, 23.6, 8.4],
)

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

R = [rand(Int8, 3, 4) for δ = 1:5]

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
    m = if length(matr) < 65
        SMatrix{size(matr)...}(matr)
    elseif length(matr) < 100_000
        matr
    else
        sparse(matr)
    end
    m
end


function makearr(n, m, d)
    @assert d <= 1.0
    on = floor(n * m * d) |> Int64
    ze = n * m - on
    coll = vcat(zeros(ze), rand(on))
    #@show ze, on, length(coll)
    c = rand(coll, n, m)
end

a = makearr(20000, 5000, 0.051);
aa = Arr(a)



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

abstract type AA{T,U<:AbstractArray} end


struct Aa{T,U<:AbstractArray} <: AA{T,U}
    a::T
    b::U

end

m1 = Matrix{Float64}(undef, 2, 3)
m2 = Matrix{Int64}(undef, 2, 3)

SM1 = sparse_or_static_mat(m1)
MM1 = sparse_or_mutstat_mat(m2)

a1 = Aa(SM1, MM1)
a1.a
m2 = spzeros(10, 2)
m1 = MMatrix{2,3}(m1)
a2 = Aa(m1, m2)


size_of_change_2_sparse = 1_000_000
function sparse_or_static_mat(
    h::Matrix{T};
    ssize=size_of_change_2_sparse,
)::AbstractMatrix where {T<:Number}
    if length(h) < 100
        return SMatrix{size(h)...}(h)
    elseif 100 <= length(h) < ssize
        return h
    else
        return sparse(h)
    end
end

function sparse_or_mutstat_mat(h::Matrix{T})::AbstractMatrix where {T<:Number}
    if length(h) < 100
        return MMatrix{size(h)...}(h)
    else
        return sparse(h)
    end
end

abstract type foo{T,U<:AbstractArray} <: AbstractMatrix{T} end
struct myfoo{T,U<:AbstractArray} <: foo{T,U}
    e_id::Int64                          # n of hyperedges (ie columns)
    v_id::Int64                          # n of nodes (rows)g
    H::T
    W::U
end
a = myfoo{Matrix{Int64},Matrix{Int64}}(1, 2, [1 2 3; 4 5 6], [1 0 1; 9 1 9])

######################################################################
using BenchmarkTools
using SuiteSparseGraphBLAS, LinearAlgebra, SparseArrays
# Standard arithmetic semiring (+, *) matrix multiplication
s = sprand(Float64, 10000, 10000, 0.05);
v = sprand(Float64, 10000, 1000, 0.1);
@btime s * v # 912.469 ms (7 allocations: 152.61 MiB)
#### 
s = GBMatrix(s);
v = GBMatrix(v);
# Single-threaded
@btime s * v # 398.856 ms (8 allocations: 231.33 MiB)
#

# Indexing
s = sprand(Float64, 100000, 10000, 0.05); # 34.059 ms (12 allocations: 7.62 MiB)
@btime s[1:10:end, end:-10:1]

s = GBMatrix(s);
@btime s[1:10:end, end:-10:1] # 6.480 ms (22 allocations: 7.61 MiB)


Aa = GBMatrix(
    [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 7],
    [2, 4, 5, 7, 6, 1, 3, 6, 3, 3, 4, 5],
    [1:12...],
)
aa = Matrix(Aa)
aa * aa
aa .^ aa
aa .^ aa - emul(Aa, Aa, ^) # elementwise exponent
Aa .^ Aa
@btime map(sin, Aa)
@btime sin.(Aa)

M = GBMatrix([[1, 2] [3, 4]])
M .+ 1

h = hhsp
Ei = h.h_edges - I
DeInv_vec = [Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)] 
DeInv = GBMatrix([eachindex(DeInv_vec)...], [eachindex(DeInv_vec)...], DeInv_vec)
andp = h.H * h.weights * DeInv * h.H' - h.nodes
##########################################################
using SparseArrays
using LuxurySparse
using BenchmarkTools

pm = pmrand(7)  # a random permutation matrix
id = IMatrix(3) # an identity matrix
@benchmark kron(pm, id) # kronecker product

Spm = pm |> SparseMatrixCSC  # convert to SparseMatrixCSC
Sid = id |> SparseMatrixCSC
@benchmark kron(Spm, Sid)    # compare the performance to the previous operation.

spm = pm |> staticize        # convert to static matrix, notice that `id` is already static.
@benchmark kron(spm, spm)    # compare performance
@benchmark kron(pm, pm)
####################

using DrWatson
@quickactivate "graphs"

includet(srcdir("HPRA.jl"))
includet(srcdir("HPRA_incidence.jl"))

###############################################################
struct z{T}
    i::T
    a::Matrix{Float64}
end

struct zz
    i::Any
    a::SubArray{Float64}
    function zz(i, a)
        new(i, view(a, 1:i, 1:i))
    end
end
z1 = z{Int64}(3, [1 2 3; 4 5 6; 7 8 9])
zz1 = zz(2, z1.a)
# z and zz are immutable, but Matices are mutable, so we can chamge them.
zz1.a[1, 1] = 3
# this changes z1.a
@assert z1.a[1, 1] == 3
typeof(z1.a)
typeof(zz1.a)
###########################################################

hyperedges = Vector{HyperEdge}()
kfold = k
# we sample E^T, ie for hyperedges not in J
new_hedges_sizes(i) = sample(diag(hg.h_edges)[kfold], i)

kfoldnodes = nodes_degree_mat(hg[:, k]) |> diag

function first_nodes(i, nodes)

    nodes_distribution = (nodes |> fweights)
    return sample(1:hg.v_id, nodes_distribution, i)

end


#while length(hyperedges) < 1
# choose the degrees of the new hyperedges according to current h_edge degrees
new_hedge_size = new_hedges_sizes(1)
#@show new_hedge_size
# choose new first nodes according to current nodes degree  using Preferential Attachment
# we dont care about the current no of nodes, only their degrees, (should we use 'unique' here?)
first_node = first_nodes(1, kfoldnodes)[1]
#@show any(==(0), first_nodes) 
#sanity
#do this for n > 1000 to chech if node sampling was OK: d should be approx nodes_distribution
#= begin) # |> unique)
# we will sample according to this distribution, 
nodes_distribution = (nodes_distribution |> fw
    ufn = unique(first_nodes)
    fnode_distr = map(i -> count(==(i), first_nodes), ufn)
    dd = Dict(zip(ufn, fnode_distr)) |> sort
    norm = dd[1]
    d = Vector{Float64}(undef, length(dd))
    for (i, k) in dd
        d[i] = dd[i] / norm * nodes_distribution[1]
    end
    @show unique(first_nodes), fnode_distr, nodes_distribution
    @show d
end =#

# Create a new h-edge
# well, all of them are going to have the same hyperedge number, but it is immaterial here.
new_he = HyperEdge(hg.e_id + 1, Int64(hg.v_id), Dict([first_node => 1]), 1.0)

# Now we must add to each hyperedge, new_hedge_size-1 nodes.
# a loop over the needed number of elements in the new h-edge 
new_Vertex = 0
while length(new_he.nodes) < new_hedge_size[1]

    new_Vertex = choose_new_vertex(hg, new_he)
    if new_Vertex == 0
        # something went wrong in the new vertex calculation. 
        # Start again 
        break
    end
    new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what
    keys(new_he.nodes)
end
if new_Vertex == 0 # go back to the Start
    continue
end

# add the new vertex
new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what

# double check: 
if new_he ∉ hyperedges
    push!(hyperedges, new_he)
    #     else
    #         # ... again start this hyperedge from the start 
    #         continue
    #     end

end

################################3
for i = 1:hh.v_id
    a = sort(gethyperedges(h_rand, 1))
    bb = sort(gethyperedges(hh, 1))

    @assert keys(aa) == keys(bb)
    #@show values(aa)
    #@show values(bb)
    #@assert 
    @assert all(values(aa) .== (values(bb)))

end
##################################

function myfunc()
    A = rand(200, 200, 400)
    maximum(A)
end
myfunc()
using Profile
@profile myfunc()
@profview myfunc()
## -------------------------------------function profile_test(n)
function profile_test(n)
    for i = 1:n
        A = randn(100, 100, 20)
        m = maximum(A)
        Am = mapslices(sum, A; dims=2)
        B = A[:, :, 5]
        Bsort = mapslices(sort, B; dims=1)
        b = rand(100)
        C = B .* b
    end
end

# compilation
@profview profile_test(1)
# pure runtime
@profview profile_test(10)

####################################################

using Transducers, Folds
using BenchmarkTools



m = rand(10000, 1000)

@btime mapreduce(x -> x > 0.5 ? x^2 : x^0.5, +, m) # 109.516 ms (1 allocation: 16 bytes)
@btime Folds.mapreduce(x -> x > 0.5 ? x^2 : x^0.5, +, m) # 31.100 ms (22 allocations: 1.75 KiB)

a="Dennis, Nell, Edna, Leon, Nedra, Anita, Rolf, Nora, Alice, Carol, Leo, Jane, Reed, Dena, Dale, Basil, Rae, Penny, Lana, Dave, Denny, Lena, Ida, Bernadette, Ben, Ray, Lila, Nina, Jo, Ira, Mara, Sara, Mario, Jan, Ina, Lily, Arne, Bette, Dan, Reba, Diane, Lynn, Ed, Eva, Dana, Lynne, Pearl, Isabel, Ada, Ned, Dee, Rena, Joel, Lora, Cecil, Aaron, Flora, Tina, Arden, Noel, and Ellen sinned."
reverse(a)