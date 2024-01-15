# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.0
#     language: julia
#     name: julia-1.10
# ---

# # HPRA paper tests
# Trying out SimpleHypergraphs.jl etc for HPRA paper
#

using DrWatson
@quickactivate "graphs"

#
# The following is implemented in `HPRA.lj` (see next cell):
#
# With $V$ the set of n nodes (or vertices) and $E$ the set of m hyperedges,
# the Degree of node v is defined as $$hyper\_degree(v) = \sum_{e ∈E,\;v ∈e} w(e)$$
#
# $Neighbors (v)$ is a set containing the one-hop neighbors of node
# $v$ (nodes of hyperedges, $v$ is part of).
#
#
#
#
# Resource Allocation  
# $$RA(x, y) = \sum_{z\; ∈ \;[ \;N (x )\;∩ \; N (y)\;]} \frac{1}{d(z)}
# $$
#
# HRA Direct
# $$ HRA_{direct} (x, y) = \sum_{e \; s.t. \; x,y \;∈\;e} \frac{1}{ δ(e) - 1} $$
#
# $$HRA_{indirect} (x, y) = \sum_{z \; ∈\; [N (x) \;∩\; N(y)]} 
#  HRA_{direct} (x,z) × \frac{1}{d(z)}  × HRA_{direct} (z, y) $$
# $$ NHAS (x,e) = \frac{1}{|e|} \sum_{y \;\in \; e} HRA(x, y) $$
#

# +
#using Pkg; pkg"add SimpleHypergraphs"
# using SimpleHypergraphs
# import Graphs
# using LaTeXStrings
# using Distributions, StatsPlots

includet(srcdir("HPRA.jl"))

# +

scenes = Dict([(0, ["FN", "TH"]),
    (1, ["TH", "JV"]),
    (2, ["BM", "FN", "JA"]),
    (3, ["JV", "JU", "CH", "BM"]),
    (4, ["JU", "CH", "BR", "CN", "CC", "JV", "BM"]),
    (5, ["TH", "GP"]),
    (6, ["GP", "MP"]),
    (7, ["MA", "GP"])]) |> sort

tor = Set()
for (i, k) in scenes
    #@show i,k
    for kk in k
        push!(tor, kk)
    end
end
ttor = Dict()
for (i, val) in enumerate(tor)
    ttor[val] = i
end


sscenes = Dict()
for (i, val) in scenes

    v = []
    for k in val
        @show i,val, k, string(ttor[k])
        push!(v, [k])
    end
    sscenes[i] = v
end

# for i in sscenes
#      print(i, sscenes[i])
# end    
ttor

# +
# n of vertices 
n_vert = 13; #length(ttor)
# n of hyperegdes
n_edges = 8 #length(scenes)

h = Hypergraph{Float64}(n_vert, n_edges);
for (i,ki) in scenes
    #println("$i => $ki")
    for k in ki
        h[ttor[k],i+1] = 1
        #@show k,ttor(k], i+1
    end
end
draw(h, HyperNetX; width=6, height=6) #, no_border=true)

# +
toy = Dict([ (1 , [1,2,3]),
(2 , [1,4,5]),
(3 , [1,5,6,7,8]) ]) |> sort

htoy = Hypergraph{Float64}(8, 3);

for (i,ki) in toy
    println("$i => $ki")
    for k in ki
        htoy[k,i] = 1
        #@show k rott[k] i+1
    end
end
draw(htoy, HyperNetX; width=5, height=5, no_border=true)
# -

a = HRA_indirect(h, 6, 5)
b = HRA_direct(h, 6, 5)
#intersect(a,b)
a,b

NHAS(h,11,7)


# test HRA direct as per fig 1 of HPRA paper
for nod in sort(h_nodes(htoy))
      hrad =  HRA_direct(htoy,1,nod)
      hrain = HRA_indirect(htoy,1,nod)
      hra   = HRA(htoy,1,nod)
    
    println("$(nod) -> hra(1,$(nod)) = $(hrad) + $(hrain) = $(hra)")
end

h_edges(htoy)

node_degree(h,4,edge_size=1)


weighted_h = deepcopy(h)

# Now we add the connection weights

weighted_h[1:3,1] .= 1
weighted_h[3,4] = 2
weighted_h[2,3] = 3
weighted_h[4,3:4] .= 4
weighted_h[5,4] = 5
weighted_h[4:5,2] .= 6
weighted_h[7,5] = 1
weighted_h[6,5] = 1
weighted_h[7,1] = 10
weighted_h

for n in h_nodes(h)
    d= node_degree(h,n)
    dw = node_degree(weighted_h,n)
    @show n, d, dw
end    




a = nodes_by_degree(h);
@show a;

nhe(h), nhv(h)

n4 = h_Neighbours(h,6)
n5 =h_Neighbours(h,13)
a=RA(h,4,5)
@show n4, n5
@show n4 ∩ n5
@show a

for i in sort(h_nodes(h))
    println("$i $(RA(h,i,4))")
end

draw(h, HyperNetX; width=5, height=5, no_border=true)

h.v2he

conductance(h, Set([1,2,3]))

hh = Hypergraph{Int64}(2, 3) #, v_meta = Vector{Union{Nothing,String}}(nothing, 2));
hh[1, 2:3] .= 1;
hh[2, 3] = 1;
hh


# ### Remember, columns are h-edges, rows are nodes.

#add_vertex!(hh)
add_hyperedge!(hh, vertices = Dict(1=>1,2=>1))
add_hyperedge!(hh)
# add a new vertex with 
add_vertex!(hh, hyperedges=Dict(1 => 1, 2 => 1))
# This added a new row at the bottom 
# if we need to add  vertex v to hedge e we just change hh[e,v] (which should be nothing). 
# add vertex 5 to edges 1 and 3
for i in (1,3)
    hh[i,5] = 1
end
hh



@show hh[1,2]  
@show nhe(hh),nhv(hh)
# add a hyperedge which contains node 2
hh[end,end] =1
hh[2,1] = 1
hh


add_vertex!(hh, hyperedges=Dict(1 => 1, 2 => 1))
hh

draw(hh, HyperNetX; width=5, height=5, no_border=true)

draw(hh, HyperNetX; width=5, height=5, no_border=false)

# ## Open some HPRA data files

# +
using MAT
hpradatadir = projectdir("HyperedgePrediction/dataset_utils/Datasets")

citeseer1 = matread( joinpath(hpradatadir, "citeseer_coreference.mat") )
citeseer2 = matread( joinpath(hpradatadir, "citeseer_cocitation.mat") )
# -

keys(citeseer1)

keys(citeseer2)

c1 = citeseer1["S"] # this is just a shortcut
c2 = citeseer2["S"]
c1[1:20,1:10]

c1

c2

Hcitecoref = Hypergraph{Float64}(size(c1)...)  # size(c1) = (1299, 626)
nonzeros = findall(!=(0.), c1) # c1 is a swallow copy of citeseer1["S"]
Hcitecoref[nonzeros] .= 1.;

# this is not very illuminating...
draw(Hcitecoref, HyperNetX; width=18, height=20, no_border=true)

# The distribution of hyperdge |e|
hydist = hyperedge_dist(Hcitecoref).vect
size(hydist), typeof(hydist)
histogram(hydist, bins=minimum(hydist):(maximum(hydist)+1), yscale=:log10,normalize=:probability) 

using StatsBase
plot(ecdf(hydist))

xx = [0:0.001:5]
plot(xx, cdf.(Weibull(), xx), label="Weibull cdf")
plot!(xx, pdf.(Weibull(), xx), label="Weibull pdf")
plot!(xx, pdf.(Exponential(), xx), label="Exp pdf")
plot!(xx, cdf.(Exponential(), xx), label="Exp cdf")

# For the time being we use the wweibull (not a good match) in order to get the results
hyfitW = fit(Weibull, hydist)
hyfit = fit(Exponential, hydist)
params(hyfit), params(hyfitW) #shape(hyfit), scale(hyfit)

# +
mybins = minimum(hydist):(maximum(hydist)+1)
histogram(hydist, bins=mybins, normalize=:pdf) #probability)

plot!(mybins, pdf.(hyfit, mybins), linewidth=4, linecolor=:black, label= "Exp")#, yscale=:log10)
plot!(mybins, pdf.(hyfitW, mybins), linewidth=4, label="Weibull")#, yscale=:log10) # linecolor=:black)
# -

# Do the same for the nodes.
nodedist = length.(Hcitecoref.v2he)
nodebins = minimum(nodedist):(maximum(nodedist)+1)
histogram(nodedist, bins=nodebins, normalize=:pdf) #, yscale=:log10) #, normalize=:probability)
# do we need a fit of some kind?
nodefitW = fit(Weibull, nodedist)
nodefit = fit(Exponential, nodedist)
params(nodefit)
plot!(nodebins, pdf.(nodefit, nodebins), linewidth=4, linecolor=:black)
plot!(nodebins, pdf.(nodefitW, nodebins), linewidth=4, linecolor=:red)


# But in the case of nodes we are interested in "preferential
# attachment, i.e., nodes with a higher degree are more
# likely to form new links. Following this, once the cardinality d of new hyperedge is determined, we choose
# the first member of the hyperedge with probability
# proportional to the node degrees.''
#

# +
pooldensity = nodedist |> unique |> sort
pool = Vector{Int}(undef,0);
# poor man's weighted distribution.
# oool contains node degrees. It contains n elements with value n, eg there are 5 fives and 7 sevens
# so that when we sample it, the probability to choose degree d will be proportional to the degree.
for i in pooldensity
    for j in 1:i
        push!(pool,i)
    end
end

# create the sampler
nodeSpl = Spl(pool)
# use it like this:
rand(nodeSpl, 2)

# +
# Get a dict degree => Vector(nodes with this degree) 
# For a particular degree we can choose a node  
nbd = nodes_by_degree(Hcitecoref)
# sanity: this must be the same lenght as the uniqy elements in nodeist 
@assert ( nodedist |> unique |> length )  == length( nbd ) 

# for degree d we can choose a random node as follows:
d = 10.
rand(nbd[d],3)
# -

# ### Algorithm 1: Hyperedge Prediction using Resource Al-location (HPRA)
#
# - Input: Hypergraph Incidence Matrix H, Node set V,
# Hyperedge Degree Distribution HDD
# - Output: Predicted Hyperedge e
#
# 1. `` Sample hyperedge degree from HDD ``
# 2. ``d ← get_degree(hyperedge_degrees, prob = H DD)``
# 3. ``// Initialize hyperedge e ``
# 4. ``e ← {} `` 
# 5. ``// Select first node using Preferential Attachment``
# 6. ``vnew = get_node(V , prob = node_degrees)``
# 7. ``e.add(v new )``
# 8. ``while size(e) < d do``
# 9. ``// Compute NHAS for remaining nodes`` (See Algo 2)
# 10.  ``scores ← NHAS( e, V )``
# 11. ``// Select a node based on NHAS``
# 12. ``v ← get_node(V , prob = scores)``
# 13. ``e.add(v)``
# 14. ``end``

# ### Algorithm 2: Node-Hyperedge Attachment Scores
# - Input: Edge e, Node set V, HRA score matrix HRA
# - Output: Node-Hyperedge Attachment Scores scores)
# 1.  ``// Initialize scores``
# 2. $scores ← zeroes(size(V))$
# 3. ``// Compute NHAS for each node in`` $V$\ $e$
# 4. ``for`` $v_i$ ``in ``$V$ ``do``
# 5. <code>&nbsp;&nbsp;</code>``if `` $v_i$ ``not in`` $e$ ``then``
# 6.  <code>&nbsp;&nbsp;&nbsp;&nbsp;</code>  ``for`` $v_j$ ``in`` $e$ ``do``
# 7.   <code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</code> $scores[i] ← scores[i] + HRA(v_i , v_j )$
# 8.  <code>&nbsp;&nbsp;&nbsp;&nbsp;</code> ``end``
# 9. <code>&nbsp;&nbsp;&nbsp;&nbsp;</code> $scores[i] ← \frac{1}{size(e)} ∗ (scores[i])$      
# 10. <code> &nbsp;&nbsp;</code>``end``
# 11.   ``end``

# ## To Be Done
#
# - P - Set of To Be Predicted Hyperegdes
# - V - Set of nodes
#
#
# For $q \geq 1, q\in \mathbb{Z} $
#
# $P_q = \{x \in 2^V : |x| = q\}$
#
#  $|P_q| =  { n \choose q }$,  where $n =|V|$ 
#  
#  (Cardinality) 

# #### A Observations -- Training set
# Random Hypergraph A with args:
# - No of nodes ~ 30
# - No of hyperedges  ~ 10-20
# - Min max of nodes in each hyperedge min ~ 1- 3 , max ~ 4-8
#
# #### B Predictions -- Test set
# To be predicted hypergraph B with args:
# - The same nodes as in A
# - No of hyperedges  ~ 3 - 6
# - Min max of nodes in each hyperedge min ~ 1-3 , max ~ 4-8

# A Hyperedge with $n$ nodes corresponds to:
# - $n$-clique (Hypergraphs)
# - $(n-1)$-simplex  (Simplicial complexes)

# +
# how many edges and nodes?

nhe(Hcitecoref), nhv(Hcitecoref)
# -

function create_new_hyperedge(hg::Hypergraph, nodeSpl::Spl)

    # Step 3 - 4: We get a hyperedge degree with the help of a sampler from the graph, before we add a new edge
    samp = hyperedge_dist(hg)
    deg_of_new_e = rand(samp)[1] # rand(samp) is a Vector w/ 1 element, so derefence

    # Get a dict degree => Vector(nodes with this degree) 
    # For a particular degree we can choose a node  
    nbd = nodes_by_degree(hg)
    
     #step 5-6 get first node
    deg_of_first_node = rand(nodeSpl) # this is also a vector w/ 1 element, so use it as in nbd[ deg_of_first_node[1] ]
    
    # get a random vertex from the vec of vertices w/ deg as found
    first_node = ( nbd[ deg_of_first_node[1]] |> rand )

    # step 1 - 2
    # So now we can add the new hyperdge, whcih contains the first node w/ weight 1.
    add_hyperedge!(hg, vertices=Dict(first_node => 1.0))
    new_he = NewHEdge(nhe(hg), [first_node])


    # a loop over the needed number of elements in the new h-edge 
    while length( new_he.nodes)  < deg_of_new_e

        #calculate scores 
        scores = calc_NHAS(hg, new_he)

        # Choose a node (which is not in new_he) w/ prob proportional to its score 
        # -> the scores are weights of the sampling. so   
        # we can take only non zero scores (since 0 dows not contribute anything)
        # and also we ignore the nodes in new_he (these are already in)
        indeces = findall(!=(0.0), scores)
        scores_view = getindex(scores, indeces)

        # choose a node
        newV_idx = sample(indeces, Weights(scores_view))
        println("Chose node $(newV_idx) w/ a score of $(scores[newV_idx])")

        # This is added to the new hyperegde and to new_he ecdfplot
        push!(new_he.nodes, newV_idx)
        hg[new_he.e_id, newV_idx] = 1. ##  or weight
        
    end
    
    return new_he
end

# Step 3 - 4: We get a hyperedge degree with the help of a sampler from the graph, before we add a new edge
samp = hyperedge_dist(Hcitecoref)
deg_of_new_e = rand(samp)[1]
#typeof(d)


create_new_hyperedge(Hcitecoref, nodeSpl)

# step 5-6 get first node
deg_of_first_node = rand(nodeSpl)
# this is a vector w/ 1 element, so use it as in
#nbd[ deg_of_first_node[1] ]
# get a random vertex from the vec of vertices w/ deg as found
first_node = ( nbd[ deg_of_first_node[1]] |> rand )


# step 1 - 2
# So now we can add the new hyperdge, whcih contains the first node w/ weight 1.
add_hyperedge!(Hcitecoref, vertices=Dict(first_node => 1.))


new_he = NewHEdge(nhe(Hcitecoref), [first_node])

# +
# So now we loop over all the nodes not in new_he and compute theri HNAS wrt new_he

function calc_NHAS_test(h::Hypergraph, new_he::NewHEdge)
    # Initialize
    nhas_scores = zeros(Float64, nhv(h))
    scores      = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff( h_nodes(h), new_he.nodes )
        sc = NHAS(h, nod, new_he.e_id)
        if sc != 0.
            nhas_scores[nod] = sc

            for j in new_he.nodes
                scores[nod] +=  HRA(h, nod, j)
            end
            scores *= 1. /length(new_he.e_id)
        end 
    end
    return scores, nhas_scores
end

function calc_NHAS(h::Hypergraph, new_he::NewHEdge)
    # Initialize
  
    scores      = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff( h_nodes(h), new_he.nodes )
        sc = NHAS(h, nod, new_he.e_id)
        if sc != 0.
            scores[nod] = sc

            # for j in new_he.nodes
            #     scores[nod] +=  HRA(h, nod, j)
            # end
            # scores *= 1. / length(new_he.e_id)
        end 
    end
    return scores
end
# -

scores, nhas_scores = calc_NHAS_test(Hcitecoref, new_he)
sscores = calc_NHAS(Hcitecoref, new_he)

# +
#unique(sscores)

# +
# This is the test 
a = [nhas_scores[i]  for i in h_nodes(Hcitecoref) ] |> unique |> sort
b = [scores[i]  for i in h_nodes(Hcitecoref) ] |> unique |> sort

@assert (a - b |> unique) == [0.]
# also 
@assert unique(nhas_scores - scores) == [0.]
@show length(scores), new_he.nodes


# +
# Choose a node (which is not in new_he) w/ prob proportional to its score 
#indeces = collect( 1:length(scores) )
# remove all  new_he vertices
#filter!(ind -> ind ∉ new_he.nodes, indeces)
# we take a view of scores that does not contain the nodes in new_he
indeces = findall(!=(0.), scores)
scores_view = getindex(scores, indeces)

#@assert 
@assert length(scores_view) == length(indeces)
length(scores_view), length(indeces)
# -

## Test
newV_idx = sample(indeces, Weights(scores_view), 100)
#sample(items, Weights(weig), 6, replace=true) |> sort
#
@assert [i for i in newV_idx if scores[newV_idx] ==0] == []

newV_idx = sample(indeces, Weights(scores_view) )
println("Chose node $(newV_idx) w/ a score of $(scores[newV_idx])")

# +
# This is added to the new hyperegde and to new_he ecdfplot
# -





# ## Visualizing a hypegraph
#
# To visualize a given hypergraph `h`, the user needs to specify two mandatory parameters:
# 1. the hypergraph `h` to draw
# 2. which method should be used to visualize `h`
#     * `GraphBased` represents each hyperedge `he` with a *fake* vertex `fv` to which each vertex `v ∈ he` is connected.
#     * `HyperNetX` renders an Euler diagram of the hypergraph where vertices are black dots and hyper edges are convex shapes containing the vertices belonging to the edge set. 

# ### A `GraphBased` visualization

# #### Vertices options

# * If `with_node_labels=true`, but `node_labels` is not specified, vertex ids will be used as their label.

SimpleHypergraphs.draw(h, 
    GraphBased; 
    width=1000, 
    height=1500,
    radius=10, #same radius for each node
    node_color = "yellow", #same color for each node
    node_stroke="orange", #same stroke for each node
    stroke_width=2, #same stroke-width value for each node
    node_opacity=0.5, #same opacity for each node
    with_node_labels=true, #wheter displaying or not node labels
    with_node_metadata_hover=true
)

# * Different radii, colors, strokes, stroke-widths, opacities and labels can be specified for each node. If one of these parameters is specified, the corresponding default value for each vertex will be ignored.

SimpleHypergraphs.draw(
    h, 
    GraphBased; 
    width=500, 
    height=500,
    radius=10, #same radius for each node
    node_color = "yellow", #same color for each node
    node_colors = ["yellow", "yellow", "yellow", "blue", "red", "red", "blue"],
    node_stroke = "orange", #same stroke for each node
    node_strokes =  ["orange", "orange", "orange", "orange", "black", "black", "black"],
    stroke_width=2, #same stroke-width value for each node
    node_opacity=0.5, #same opacity for each node
    with_node_labels=true, #whether displaying or not node labels
    node_labels=["A","B","C","D","E","F","G"],
    with_node_metadata_hover=true,
)

# * If `with_node_weight=true`, each vertex weight within the hyperedges it belongs to will be displayed.

SimpleHypergraphs.draw(
    h, 
    GraphBased; 
    width=500, 
    height=500,
    radius=10, #same radius for each node
    node_color = "yellow", #same color for each node
    node_stroke="orange", #same stroke for each node
    stroke_width=2, #same stroke-width value for each node
    node_opacity=0.5, #same opacity for each node
    with_node_labels=true, #whether displaying or not node labels
    node_labels=["A","B","C","D","E","F","G"],
    with_node_metadata_hover=true,
    with_node_weight=true
)

# #### Hyperedges options

draw(
    h, 
    GraphBased; 
    width=500, 
    height=500,
    radius=10, #same radius for each node
    node_color = "yellow", #same color for each node
    node_stroke="orange", #same stroke for each node
    stroke_width=2, #same stroke-width value for each node
    node_opacity=0.5, #same opacity for each node
    with_node_labels=true, #whether displaying or not node labels
    with_node_metadata_hover=true,
    with_node_weight=true, #whether displaying vertices metadata on mouse hover
    he_colors=["green", "blue", "red", "yellow","black"], #hyperedges colors
    with_he_labels=true, #whether displaying or not hyperedge labels
    he_labels=["a","b","c","d"], #hyperedges labels
    with_he_metadata_hover=true #whether displaying hyperedges metadata on mouse hover
)

# **SimpleHypergraphs** integates the Python library **HyperNetX** to let the user visualize a hypergraph `h` exploiting an Euler-diagram visualization. For more details, please refer to the library [HyperNetX][https://github.com/pnnl/HyperNetX).

# There are many options for `Hypergraph` plotting. Type `?draw` to see them all.

# +
#? draw # press Ctrl+Enter to see documentation for `draw`
# -




# ## Bipartite View of the hypergraph
# The type `BipartiteView` represents a non-materialized view of a bipartite representation hypergraph `h`. Note this is a view - changes to the original hypergraph will be automatically reflected in the view.
#
# The bipartite view of a hypergraph is suitable for processing with the `LightGraphs.jl` package.
#
# Several LightGraphs methods are provided for the compability.

b = BipartiteView(h)

# The `BipartiteView` provide LightGraphs.jl compability. 

supertype(typeof(b))

# We add here a edge to a parent Hypergraph of a bisection view. Note that this change will be reflected in the bipartite view

add_vertex!(h)
add_hyperedge!(h)



# This graph can be plotted using `LightGraphs` tools. 

# +
using GraphPlot
using Graphs
nodes, hyperedges = size(h)
nodes_membership = fill(1, nodes)
hyperedges_membership = fill(2, hyperedges)

membership = vcat(nodes_membership, hyperedges_membership)

nodecolor = ["lightseagreen", "orange"]
#membership color
nodefillc = nodecolor[membership]

gplot(b, nodefillc=nodefillc, nodelabel=1:nv(b), layout=circular_layout)
# -

# The functionality of `LightGraphs` can be used directly on a bipartite view of a hypergraph. 

Graphs.a_star(b, 1, 3)

#number of vertices
nv(b)

#number of edges
ne(b)

#neighbors
sort(collect(outneighbors(b,5)))

#neighbors
sort(collect(inneighbors(b,9)))

#shortest path - it does not consider the nodes associated with a hyperedge
shortest_path(b,1,4)

# ## Twosection View of the hypergraph
# Represents a two section view of a hypergraph `h`. Note this is a view - changes to the original hypergraph will be automatically reflected in the view.
#
# The bipartite view of a hypergraph is suitable for processing with the `LightGraphs.jl` package.
#
# Several LightGraphs methods are provided for the compability.
#
#  Note that the view will only work correctly for hypergraphs not having overlapping hyperedges. To check
#   whether a graph has overlapping edges try has_overlapping_hedges[h) - for such graph you need to fully
#   materialize it rather than use a view. This can be achieved via the get_twosection_adjacency_mx[h) method.

# This condition is required for an unmaterialized `TwoSectionView` representation of a hypergraph to make sense
@assert SimpleHypergraphs.has_overlapping_hedges(h) == false

t = TwoSectionView(h)

gplot(t, nodelabel=1:nv(t))

#number of vertices
nv(t)

#number of edges
ne(t)

#neighbors
sort(collect(outneighbors(t,5)))

#neighbors
sort(collect(inneighbors(t,1)))

#shortest path 
shortest_path(t,1,5)

# ## Community detection in hypergraphs

# Let us consider the following hypergraph

# +
h = Hypergraph{Float64}(8,7)
h[1:3,1] .= 1.5
h[3,4] = 2.5
h[2,3] = 3.5
h[4,3:4] .= 4.5
h[5,4] = 5.5
h[5,2] = 6.5
h[5,5] = 5.5
h[5,6] = 6.5
h[6,7] = 5.5
h[7,7] = 6.5
h[8,7] = 6.5
h[8,6] = 6.5

h
# -

# Let us search for communities in the hypergraph `h`

# +
best_comm = findcommunities(h, CFModularityCNMLike(100))

display(best_comm.bm)

display(best_comm.bp)
# -

# And now we visualize them in 2-section view

# +
t = TwoSectionView(h)

function get_color(i, bp)
    color = ["red","green","blue","yellow"]
    for j in eachindex(bp) #1:length(bp)
        if i in bp[j]
            return color[j]
        end
    end
    return "black"
end

gplot(t, nodelabel=1:nv(t), nodefillc=get_color.(1:nv(t), Ref(best_comm.bp) ))



# -






