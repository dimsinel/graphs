# # HPRA paper tests
# Trying out SimpleHypergraphs.jl etc for HPRA paper
#
using DrWatson
@quickactivate "graphs"

includet(srcdir("HPRA.jl"))
includet(srcdir("HPRA_incidence.jl"))


h_rand = random_model(7, 20)
# --- or
h_rand = random_preferential_model(7, 0.1)
@show size(h_rand)
a_rand_hg = (replace(h_rand, true => 1) |> Hypergraph)
display(a_rand_hg)# @show find_empty_nodes(a_rand_hg)
# +
hh = myHyperGraph(h_rand)
A(hh)
Incidence(h_rand)


length.(h_rand.he2v)
hh.
v_id = 3
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





hh = Incidence(h_rand) #( replace(h_rand, nothing=>0., true=>1.) |> Matrix)
h = h_rand
D = nodes_degree_mat(h)

E = hyperedges_degree_mat(h)
W = hyper_weights(h)
A = h * W * h' - D
Ei = E - I

DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in 1:size(Ei, 1)] |> Diagonal)
sum(DeInv * Ei)

A_ndp = h * W * DeInv * h' - D

dim1 = size(a_rand_hg, 1)
hradir = Matrix{Union{Nothing,Float64}}(undef, dim1, dim1);
for i in 1:dim1
    for j in 1:dim1
        hradir[i, j] = HRA_direct(a_rand_hg, i, j)


    end
end
hradir


aa = myHyperGraph([1, 2, 5], [5, 2, 0, 8])









using MAT
hpradatadir = projectdir("HyperedgePrediction/dataset_utils/Datasets")

citeseer1 = matread(joinpath(hpradatadir, "citeseer_coreference.mat"))
citeseer2 = matread(joinpath(hpradatadir, "citeseer_cocitation.mat"))
# -

c1 = citeseer1["S"] # this is just a shortcut
c2 = citeseer2["S"]


Hcitecoref = Hypergraph{Float64}(size(c1)...);  # size(c1) = (1299, 626)
nonzeros = findall(!=(0.0), c1) # c1 is a swallow copy of citeseer1["S"]
Hcitecoref[nonzeros] .= 1.0;



@time for i in 1:10
    ss = h_Neighbours_mat(Hcitecoref, i)
    #ss2 = h_Neighbours(Hcitecoref, i)
    #@assert setdiff(ss2,ss) == Set{Any}() 
    #if length(ss) != 0 || length(ss) != 0
    #@show i, ss, ss2
    #end
end
# The distribution of hyperdge |e|
# hydist = hyperedge_dist(Hcitecoref).vect
# size(hydist), typeof(hydist)
# histogram(hydist, bins=minimum(hydist):(maximum(hydist)+1), yscale=:log10,normalize=:probability) 



using StatsBase
#plot(ecdf(hydist))




# But in the case of nodes we are interested in "preferential
# attachment, i.e., nodes with a higher degree are more
# likely to form new links. Following this, once the cardinality d of new hyperedge is determined, we choose
# the first member of the hyperedge with probability
# proportional to the node degrees.''
#

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
    tot_n = 10
    tot_nodes = 0
    nn = 1
    part_idx = 600
    onefold = size(Hcitecoref)[2] - part_idx
    for ll in 1:nn
        tt = @elapsed begin
            setOfEdges = []
            hg = Hypergraph(Hcitecoref[:, 1:part_idx])
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


foldem(Hcitecoref, fold_k)


