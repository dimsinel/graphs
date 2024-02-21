# # HPRA paper tests
# Trying out SimpleHypergraphs.jl etc for HPRA paper
#
using DrWatson
@quickactivate "graphs"

includet(srcdir("HPRA.jl"))
includet(srcdir("HPRA_incidence.jl"))

begin

    h_rand = random_model(7, 20)
    # --- or
    h_rand = random_preferential_model(7, 0.1)
    @show size(h_rand)
    h_rand[5:7, 1:(end-5)] .= nothing
    a_rand_hg = (replace(h_rand, true => 1) |> Hypergraph)
    display(a_rand_hg)# @show find_empty_nodes(a_rand_hg)


    hh = myHyperGraph(h_rand)
    A(hh)
    A_ndp(hh)
    Incidence(h_rand);
end

h_rand

begin
    using MAT
    hpradatadir = projectdir("HyperedgePrediction/dataset_utils/Datasets")
    matfiles = Dict()
    alphabetic = ['j', 'k', 'b', 'a', 'd', 'c', 'e']
    matfiles = Dict(alphabetic .=> readdir(hpradatadir,))

    function readfilename(hpradatadir, f)
        matread(joinpath(hpradatadir, f))["S"]
    end
end
#= 
matfiles
Dict{Char, String} with 7 entries:
  'j' => "aminer_cocitation.mat"
  'a' => "citeseer_coreference.mat"
  'd' => "cora_cocitation.mat"
  'k' => "aminer_coreference.mat"
  'c' => "cora_coreference.mat"
  'e' => "dblp.mat"
  'b' => "citeseer_cocitation.mat"
=#
begin
    c1 = readfilename(hpradatadir, matfiles['j'])

    Hcitecoref = Hypergraph{Float64}(size(c1)...)  # size(c1) = (1299, 626)
    nonzeros = findall(!=(0.0), c1) # c1 is a swallow copy of citeseer1["S"]
    Hcitecoref[nonzeros] .= 1.0
end


    h = Hcitecoref
    H = Incidence(h)
    D = nodes_degree_mat(h)
    E = hyperedges_degree_mat(h)
    eid = size(E, 1)
    nid = size(D, 1)
    W = hyper_weights(h)
begin
    d = Dict{Int64,Set{Int64}}()
    for i in 1:nid
        d[i] = h_Neighbours(h, i)
    end

    Ei = E - I
    DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in eachindex(view(Ei, :, 1))] |> Diagonal)
end

begin
    @show sizeof(H), size(H)
    @show sizeof(D), size(D)

    @show sizeof(E), size(E)
    @show sizeof(W), size(W)
    @show sizeof(DeInv), size(DeInv)
end

begin
    Andp = sparse(Float32.(D))
    temporary =Float32.(W * DeInv)
    @show isdiag(temporary)
    #Dtemp = Matrix{Float64}(undef, size(H, 1), size(temporary, 2))
    Dtemp = spzeros(Float32, size(H, 1), size(temporary, 2))
    Dtemp = Matrix{Float32}(undef, size(H, 1), size(temporary, 2)) |> sparse_or_mutstat_mat
   
    @show typeof(Dtemp)
    Dtemp = H*temporary
    @show  typeof(Dtemp)
    Andp = Dtemp*H' - Andp
end
# begin
#     Andp1 = sparse(Float64.(D))
#     Dtemp = Matrix{Float64}(undef, size(H, 1), size(temporary, 2)) |> sparse_or_mutstat_mat
#     # Dtemp = H * temporary
#     mul!(Dtemp, H, temporary, 1.0, 0)
#     # @show typeof(Dtemp)
#     # Andp = Dtemp * H' - Andp
#     mul!(Andp1, Dtemp, transpose(H), 1.0, -1.0)
# end

Andp
HG1 = myHyperGraph(eid, nid, H, D, E, W, d, Andp)

HG = myHyperGraph(Hcitecoref)


begin
    hyperg = h_rand
    #citecoref #h_rand
    myhyperg = hh
    fold_k = 5

    #= kf = kfolds(size(hyperg)[2], fold_k)
    size.(kf)
    size.(kf[1]), size.(kf[2])
    size(hyperg) =#
    # av_f1 = foldem(myhyperg, fold_k)




    
    cv = collect(kfolds(myhyperg.e_id, fold_k))
    av_f1_scores = Float64[]
end
begin
    (k,j) = (cv[1][1], cv[2][1])
    # try this out:
    hhyperg = deepcopy(hyperg)
    #remove_hyperedge!(hhyperg, k)

end

#for (k, j) in zip(cv[1], cv[2])
begin 
    (k, j) = collect(zip(cv[1], cv[2]))[end]
    kept_hedges = find_connected_he(myhyperg, (k, j))
    hhg = 
    if typeof(myhyperg.H) <: SparseMatrixCSC 
        myhyperg.H[:, k]
    else
        replace(myhyperg.H[:, k], 0 => nothing)  |> myHyperGraph
    end
end
#new_Hedges = create_new_hyperedge(hhg, n=length(kept_hedges))
new_Hedges = create_new_hyperedge(hhg, n=length(j))
 

length(av_f1)
@info "length av_fi: $(length(av_f1)). Mean av_f1 $(round(mean(av_f1),digits=3)) ± $(round(std(av_f1),digits=3))"
# Compare w/ 3rd col of Table 2
# (a)  length av_fi: 5. Mean av_f1 0.183 ± 0.014
# (a)  length av_fi: 10. Mean av_f1 0.112 ± 0.024

# (b)  length av_fi: 4. Mean av_f1 0.244 ± 0.034  (The first of 5 kfolds is 0)
# (b)  length av_fi: 10. Mean av_f1 0.188 ± 0.053

# (c)  length av_fi: 5. Mean av_f1 0.132 ± 0.011
# (c)  length av_fi: 10. Mean av_f1 0.073 ± 0.02

# (d)  length av_fi: 5. Mean av_f1 0.239 ± 0.026
# (d)  length av_fi: 10. Mean av_f1 0.176 ± 0.033

# (e)   length av_fi: 5. Mean av_f1 0.197 ± 0.011

exit()


m =HRA_Matrix(myhyperg)

function HPRA_CHS(hh::myHyperGraph, fold_k)

    cv = collect(kfolds(hh.e_id, fold_k)) # breaks  1:e_id in fold_k grpups for cross validation

    av_f1_scores = []
    for (k, j) in zip(cv[1], cv[2])
        # k is E^T, the training set and j the 'missing' set, E^M
        # check E^M for disconnected vertices. Return either the folded 
        # iterator if no such disconnected nodes are found, or 
        # the corrected iterator over which we are going to cross validate
        kept_hedges = find_connected_he(hh, (k, j))
        if kept_hedges == []
            continue # no hyperedges can be predicted in this fold
        end
        @info kept_hedges
    end
end

HPRA_CHS(myhyperg, 5)


exit()

cv = collect(kfolds(myhyperg.e_id, fold_k))

#n_loops = (cv[1] .|> length |> length)
av_f1_scores = []
(k, j) = collect(zip(cv[1], cv[2]))[1]
# k is E^T, the training set and j the 'missing' set, E^M
# check E^M for disconnected vertices. Return either the folded 
# iterator if no such disconnected nodes are found, or 
# the corrected iterator over which we are going to cross validate
onefold = find_connected_he(myhyperg, (k, j))
#miss =  
@info onefold
hhg = replace(myhyperg.H[:, k], 0 => nothing) |> Hypergraph |> myHyperGraph

# Now create new h-edges for the kfold E^T hgraph, that later we will 
# compare to onefold E^M edges. 

new_Hedges = create_new_hyperedge(hhg, n=length(j))

new_Hedges_mat = create_mat(new_Hedges)

fs = calc_av_F1score_matrix(new_Hedges_mat, myhyperg.H[:, onefold])
push!(av_f1_scores, fs)
println("fs = $(fs)")
println("###"^20)







nn = 40
HHG = Hcitecoref[:, 1:(end-nn)] |> Hypergraph |> myHyperGraph
begin
    new_Hedges = create_new_hyperedge(HHG, n=nn)
    Ep = create_mat(new_Hedges) .|> Int64
    Em = Hcitecoref[:, (end-nn+1):end] |> Hypergraph |> myHyperGraph
    EMb = Em.H .|> Int64
    a, b, c = calc_av_F1score_matrix(Ep, EMb)

    sum(b), sum(c)
end
argmaxg_prime_sum(b)
argmaxg_sum(b)


exit()




new_Hedges = create_new_hyperedge(HG, n=1000)
# 
histogram(map(x -> length(x.nodes), new_Hedges))
nodedist = []
foreach(new_Hedges) do x
    push!(nodedist, keys(x.nodes)...)
end
nds = unique(nodedist) |> sort
# nds does not contain all nodes 
missing_nodes = setdiff(1:1299, nds)
# why? 
missing_node_degs = Dict()
for i in missing_nodes
    missing_node_degs[i] = sum(HG.H[i, :])
    @show sum(HG.H[i, :])
end
count(==(1), values(missing_node_degs))
count(==(2), values(missing_node_degs))
count(==(3), values(missing_node_degs))
#@assert sum(HG.H[i,:]) == 1
unique(values(missing_node_degs))

####################3333
## Importang
## hyperedges need not be unique.
## hh has hh.e_id h-edges
@show hh.e_id
# how many of these are unique?
@show unique(eachcol(hh.H))
############################################

exit()

function v_neigh(h::myHyperGraph, node)

    intermediate = h.H[:, findall(==(1), h.H[node, :])]
    # this produces a vector w/ zeros at rows where the nodes are not neighbours of node 
    idxs = any(==(1), intermediate, dims=2)

    return findall(==(1), idxs[:, 1])

end




d = Dict{Int,Set{Int}}()
@time for i in 1:nhv(Hcitecoref)
    d[i] = h_Neighbours(Hcitecoref, i)
end

dd = Dict{Int,Vector{Int}}()
@time for i in 1:nhv(Hcitecoref)
    dd[i] = v_neigh(HG, i) # much slower
end



## dd and d are identical, except that i ∈ dd[i], but i ∉ d[i]
foreach(dd) do (i, vals)
    no_i = collect(d[i]) |> deepcopy
    push!(no_i, i) |> sort!
    # @show i, (sort(dd[i]) - no_i |> sum) 
    @assert length(sort(dd[i])) - length(sort(collect(d[i]))) == 1
    @assert (sort(dd[i]) - no_i |> sum) == 0
end





length.(h_rand.he2v)
#this gives error
hh.v_id = 3
size(h_rand)

function fill_HRA_dir(h::Hypergraph)
    hradir = Matrix{Union{Nothing,Float64}}(undef, nhv(h), nhv(h))
    for i in 1:nhv(h)
        for j in 1:nhv(h)
            hradir[i, j] = HRA_direct(h, i, j)
        end
        hradir[i, i] = 0.0
    end
    return hradir
end

function fill_HRA(h::Hypergraph)
    hradir = Matrix{Union{Nothing,Float64}}(undef, nhv(h), nhv(h))
    for i in 1:nhv(h)
        for j in 1:nhv(h)
            hradir[i, j] = HRA(h, i, j)
        end
        hradir[i, i] = 0.0
    end
    return hradir
end

function fill_HRA(h::myHyperGraph)
    hradir = Matrix{Union{Nothing,Float64}}(undef, h.v_id, h.v_id)
    for i in 1:h.v_id
        for j in 1:h.v_id
            #@show i, j
            hradir[i, j] = HRA(h, i, j)
        end
        hradir[i, i] = 0.0
    end
    return hradir
end

@time Hra_direct = fill_HRA_dir(Hcitecoref);# this is only HRA_direct
@time hra = fill_HRA(Hcitecoref); # much slower 
@time hramat = fill_HRA(HG); # this is more than 6 times faster 

@time HG = myHyperGraph(Hcitecoref)

function h_Neighbours_mat(hg::Hypergraph, v_id::Int)
    # this is slower than h_Neighbours
    neigh_hedges = findall(==(1), hg[v_id, :])
    nodesS = Set()
    for h in neigh_hedges
        nodes = findall(==(1), hg[:, h])
        #@show nodes
        push!(nodesS, nodes...)
    end
    return nodesS
end


new_Hedges = create_new_hyperedge(hh, n=1)
new_Hedges = create_new_hyperedge(HG, n=100)


@time NHAS(HG, 2, 1)  # faster
@time NHAS(Hcitecoref, 2, 1)

@time calc_NHAS(Hcitecoref, new_Hedges[1]) |> sum
@time calc_all_NHAS(HG, new_Hedges[1]) |> sum # much faster


using StatsBase

# +
nodedist = length.(Hcitecoref.v2he)
pooldensity = nodedist |> unique |> sort
pool = Vector{Int}(undef, 0);
# poor man's weighted distribution.
# oool contains node degrees. It contains n elements with value n, eg there are 5 fives and 7 sevens
# so that when we sample it, the probability to choose degree d will be proportional to the degree.
for i in pooldensity
    for j in 1:i
        push!(pool, i)
    end
end

# create the sampler
nodeSpl = Spl(pool)
# use it like this:
rand(nodeSpl, n=2)

# +
# Get a dict degree => Vector(nodes with this degree) 
# For a particular degree we can choose a node  
nbd = nodes_by_degree(Hcitecoref)
# sanity: this must be the same lenght as the uniqy elements in nodeist 
@assert (nodedist |> unique |> length) == length(nbd)

# for degree d we can choose a random node as follows:
d = 10.0
rand(nbd[d], 3)

using Folds, Random

begin
    Random.seed!(3)
    tott = 0
    totheg = 0
    tot_n = 1000
    tot_nodes = 0
    nn = 10
    part_idx = 600
    onefold = size(Hcitecoref)[2] - part_idx
    for ll in 1:nn
        tt = @elapsed begin
            setOfEdges = []
            hg = Hcitecoref[:, 1:part_idx] |> Hypergraph |> myHyperGraph
            new_Hedges = create_new_hyperedge(hg, n=onefold)
            foreach(new_Hedges) do x
                h_edge_in_cont(setOfEdges, x)
            end
            #@show size(setOfEdges), size(new_Hedges), size(Hcitecoref[:, (part_idx+1):end])
            fs = calc_av_F1score_matrix(new_Hedges, Hcitecoref[:, (part_idx+1):end])
        end
        nnodes = length(setOfEdges[1].nodes)
        tot_nodes += nnodes * length(setOfEdges)
        println("$(ll) -- After $(tot_n) tries, got $(length(setOfEdges)) different h-edges w/ $(nnodes) nodes, $(round(tt, digits=3)) sec)")
        tott += tt
        totheg += length(setOfEdges)
    end

    println("$(tot_n) loops of $(nn): total time $(round(tott, digits=2)) ")
    println("   $(round(tott/totheg, digits=2)) per hedge, $(round(tott/tot_nodes, digits=2)) per node ")
end







fold_k = 5
kf = kfolds(size(Hcitecoref)[2], fold_k)
size.(kf)
size.(kf[1]), size.(kf[2])
size(Hcitecoref)

HG
foldem(HG, fold_k)


v = [1, 2, 5, 3, 3, 4, 5]
ff = [1, 3]
found = findall(x -> x ∈ ff, v)




@debug "Verbose debugging information.  Invisible by default"
@info "An informational message"
@warn "Something was odd.  You should pay attention"
@error "A non fatal error occurred"


function rrr()
    a = rand(1:100, 100)
    a = reshape(a, (2, 50))
    d1 = maximum(a, dims=1) |> mean
    d2 = maximum(a, dims=2) |> mean
    return d1, d2
end

rrr()
scatter([rrr() for _ in 1:100000])
d1 = []
d2 = []
foreach(1:1000) do x
    x = rrr()
    push!(d1, x[1])
    push!(d2, x[2])
end
mean(d1), mean(d2)
std(d1), std(d2)