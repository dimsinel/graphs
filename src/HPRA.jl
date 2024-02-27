# using DrWatson
# @quickactivate "graphs"

using SimpleHypergraphs
import Graphs
using Distributions, StatsPlots
using Printf
using Folds
####################################################
# module HPRA

# export myHyperGraph, HyperEdge, SPL
# export A, Andp, calc_all_NHAS, calc_av_F1score_matrix, choose_new_vertex,
#     create_mat, create_new_hyperedge,
#     find_all_empty_nodes, find_connected_he, foldem
# h_edge_in_cont, HRA, HRA_direct, HRA_indirect,
# hyperedges_degree_mat, hyper_weights,
# Incidence, NHAS, node_density, nodes_degree_mat,
# includet("HPRA.jl")
# includet("HPRA_incidence.jl")

# end

##########################################################


using MLJ
measures("FScore")

# Gepmetrical mean f1 score (β=1) Dunno what levels are supposed to be ;^P
m = FScore(levels = [0, 1], checks = false)

####################################################

# used in constructing sammples of hyperegde degrees to choose from
struct Spl <: Sampleable{Univariate,Discrete}
    vect::Vector{T} where {T<:Real}
end

####################################################
function Base.rand(spl::Spl; n = 1)
    l = length(spl.vect)
    spl.vect[rand(1:l, n)]
end;

#################################################################33

# convenince struct 
mutable struct HyperEdge{T}
    e_id::Int64          # hyperedge number, eg the column no in Hypergraph's incidence matrix
    v_size::Int64        # the no of vertices the graph of this hedge contains, ie the max length of nodes
    nodes::Dict{Int64,T} # node id => name or whatever
    weight::Float64      # all nodes in a hyper edge carry the same weight
end

# for now weight is set to 1
HyperEdge() = HyperEdge{Int64}(0, 0, Dict{Int64,Float64}(), 1.0)
HyperEdge{T}(e::Int64, v::Int64) where {T} = HyperEdge(e, v, Dict{Int64,T}(), 1.0)
HyperEdge{T}(e::Int64, v::Int64, w::Float64) where {T} = HyperEdge(e, v, Dict{Int64,T}(), w)
HyperEdge{T}(e::Int64, v::Int64, d::Dict{Int64,T}, w::Float64) where {T} =
    HyperEdge(e, v, d, w)

# This one copies hyperedge e_idx from a hypergraph 
function HyperEdge{T}(
    h::Hypergraph{T,Nothing,Nothing,Dict{Int64,T}},
    e_idx::Int64,
) where {T}
    hedge = HyperEdge{T}(e_idx, size(h, 1))
    for j in eachindex(h[:, e_idx])
        if !isnothing(h[j, e_idx])
            hedge.nodes[j] = h[j, e_idx]
        end
    end
    return hedge
end
## ====================================== ##
import Base: ==

function ==(h1::HyperEdge, h2::HyperEdge)

    cond1 = h1.e_id == h2.e_id
    cond2 = h1.v_size == h2.v_size
    cond3 = h1.nodes == h2.nodes
    cond4 = h1.weight == h2.weight

    cond1 && cond2 && cond3 && cond4

end
#-----------------------------------------------------------

Base.hash(x::HyperEdge, h::UInt) = hash(:HyperEdge, hash(x.nodes, h))

#############################################################

function h_edge_in_cont(set_of_h_edges, h_edge)
    """
    set_of_h_edges is a collection of hypegedges, h_edge is a hyperedge
    Returns a set of unique h-edges, ie a set of Hyperedges. 
    For some reason the Set builtin does not work as naively expected
    """

    if any(==(h_edge), set_of_h_edges)
        return set_of_h_edges
    else
        push!(set_of_h_edges, h_edge)
    end
    return set_of_h_edges
end

#################################################################33


function HE2Vect(h::HyperEdge, dim::Int64)
    """
    Gets a vector of dimension dim out of a hyperedges' dict (called 'nodes')
    if nodes = Dict([1=>1., 3=>1.])  and dim = 4, we get [1., 0., 1., ]
    """
    hVect = Vector{Union{Nothing,Float64}}(undef, dim)
    # kk = collect(keys(h.nodes))
    # vv = collect(values(h.nodes))
    map(dd -> hVect[dd[1]] = dd[2], collect(h.nodes))
    replace!(hVect, nothing => 0.0)

    return hVect
end


#################################################################33

using Pipe: @pipe
find_empty_nodes(mat) =
    @pipe replace(mat, nothing => 0) |> reduce(+, _, dims = 2) |> any(==(0), _)
find_all_empty_nodes(mat) =
    @pipe replace(mat, nothing => 0) |> reduce(+, _, dims = 2) |> findall(==(0), _)

#################################################################33

using MLJBase, MLJ, MLUtils

struct HypergraphClassifier <: MLJBase.Probabilistic end

# fit returns the result of applying create_new_hyperedge on input X:
function MLJBase.fit(model::HypergraphClassifier, verbosity, X, y)
    fitresult = create_new_hyperedge
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# `predict` returns the passed fitresult (pdf) for all new patterns:
# MLJBase.predict(model::HypergraphClassifier, fitresult, Xnew) =
#     [fitresult for r in 1:nrows(Xnew)]

#################################################################
#
function SimpleHypergraphs.add_hyperedge!(hg::Hypergraph, newhedge::HyperEdge)
    """
    An extension wnich accepts HyperEdge as argument
    For the time being only weights == 1.0 are implemented
    """
    if newhedge.e_id <= length(hg.he2v)
        #this is not an add, issue a warning
        println(
            "Cannot add hyperedge $(newhedge.e_id), Gypergraph has $(size(hg.he2v)) h-edges. Returning",
        )
        return false
    end

    if newhedge.e_id != size(hg)[2] + 1
        println(
            "Cannot add hyperedge $(newhedge.e_id) to hypergraph: It contains $(size(hg)[2]) hyperdges.",
        )
        return false
    end

    add_hyperedge!(hg, vertices = newhedge.nodes)
    return true
end

#############################################################
function node_degree(h::Hypergraph, v_id::Int; edge_size::Int = 1)
    """Degree of a node: it is the number of edges it is contained within. 
    The optional edge_size parameter places a restriction on the size of the edges 
    you consider (default edge_size=1). The degree function looks 
    for all edges of size ≥  edge_size.
    """
    degr = 0.0
    eds = gethyperedges(h, v_id)
    for (e, w) in eds
        if w == 0.0
            continue
        end
        #println("vertex $v_id is in edge $e with weight $(w):")
        vs = getvertices(h, e)
        if length(vs) >= edge_size
            #     if v == v_id
            #         continue
            #     end
            #     println("  $v  $ww") 
            # end
            #if length(eds) >= edge_size
            degr += w

        end
    end
    if degr > eps(Float64)
        return degr
    end
    return nothing
end


#################################################################33
function edge_degree(h::Hypergraph, e_id::Int) #; edge_size::Int = 1)
    """Degree of a hyperedge e, its degree is defined as δ (e) = |e|
    """

    vs = getvertices(h, e_id)
    return length(vs)
end


#####################################################################
function h_edges(h::Hypergraph)
    # size(h) returns (n_vertices, n_edges)
    # h.v2he is a vector of the hyperedges ie , it does what the following loop woudl do: 
    # for ed_n in 1:size(h)[1] #gethyperedges(h,v_id)
    #     # gethyperedges(h,ed_n) returns the dict of the edges 
    #     # that contain vert er_n with the respecthve weights
    #     eds =  gethyperedges(h,ed_n)
    #     @show  eds  
    # end
    res_edg = []
    #res_vert = [];
    for dct in h.v2he
        #@show dct
        foreach(x -> push!(res_edg, x.first), dct)

    end
    #@show length(res_vert), sort(unique(res_edg))
    return res_edg |> unique |> sort
end

###########################################################3
function h_nodes(h::Hypergraph)
    # size(h) returns (n_vertices, n_edges)
    # h.v2he is a vector of the hyperedges ie , it does what the following loop woudl do: 
    # for ed_n in 1:size(h)[1] #gethyperedges(h,v_id)
    #     # gethyperedges(h,ed_n) returns the dict of the edges 
    #     # that contain vert er_n with the respecthve weights
    #     eds =  gethyperedges(h,ed_n)
    #     @show  eds  
    # end
    res_vert = Vector{eltype(keys(h.he2v))}()
    for dct in h.he2v
        #@show dct
        #if number_format
        foreach(x -> push!(res_vert, x.first), dct)
        #     else
        #         foreach(x->push!(res_vert, ttor[x.first]), dct)
        #     end
    end
    #@show length(res_vert), sort(unique(res_edg))
    return res_vert |> unique
end


###########################################
using LaTeXStrings
function RA(h::Hypergraph, x::Int, y::Int; n_commmon_edges::Int = 1, edge_size::Int = 1)
    L"""
 #   Resource Allocation  of 2 *not directly connected* nodes x, y is   
    the \sum_{z∈ [N(x) ∩ N(y)]} \frac{1}{d(z)},
    where N(x) the neighbours of x, and d(x) the degree of x.
    """

    nx = h_Neighbours(h, x) #, n_commmon_edges=n_commmon_edges)
    if y in nx
        return nothing
    end

    ny = h_Neighbours(h, y, n_commmon_edges = n_commmon_edges)
    ra = 0.0
    for z in nx ∩ ny
        ra += 1 / node_degree(h, z, edge_size = edge_size)
        #println("---- ",z, "  ",node_degree(h, z, edge_size= edge_size) )
    end
    return ra
end


function HRA_direct(
    h::Hypergraph,
    x::Int,
    y::Int;
    n_commmon_edges::Int = 1,
    edge_size::Int = 1,
)
    L"""
    Direct part of Hyper Resource Allocation.   
    HRA_{direct} (x, y) = \sum_{e, s.t. x,y ∈e} 1/ [δ(e) - 1] 
    (is this equal to A_ndp?) 
    """
    if x == y
        return nothing
    end
    # first we need the hedges that include both x an y
    edsx = (gethyperedges(h, x) |> keys)
    edsy = (gethyperedges(h, y) |> keys)

    common_edges = intersect(edsx, edsy)
    if common_edges == []
        return nothing
    end

    hrad = 0.0
    for ed in common_edges
        hr = edge_degree(h, ed) - 1
        hrad += 1 / hr

    end
    return hrad
end

function HRA_indirect(
    h::Hypergraph,
    x::Int,
    y::Int;
    n_commmon_edges::Int = 1,
    edge_size::Int = 1,
)
    L"""
        Inirect part of Hyper Resource Allocation.   
        HRA_{indirect} (x, y) =  \sum_{z ∈N (x) ∩ N(y)} 
        HRA_{direct} (x,z) × \frac{1}{d(z)}  × HRA_{direct} (z, y) 
    """

    # The intersection of neighbours below, by construction does not contain x or y, unless x = y
    if x == y
        return nothing
    end
    # first we need the neighborhoods of x and y
    Nx = h_Neighbours(h::Hypergraph, x)
    #setdiff!(Nx, [y]) 
    Ny = h_Neighbours(h::Hypergraph, y)

    #zs = intersect(Nx, ∩ Ny)

    zs = Nx ∩ Ny

    hra_ind = 0.0
    for z in zs

        temp = node_degree(h, z)
        # this may occasionally be nothing (eg if nodes row are 0)
        if isnothing(temp)
            continue
        end
        temp = HRA_direct(h, x, z) / temp
        temp *= HRA_direct(h, z, y)

        hra_ind += temp


    end
    return hra_ind
end

#################################################3

function HRA(h::Hypergraph, x::Int, y::Int; α = 0.5)
    if x == y
        return nothing
    end
    temp = HRA_direct(h, x, y)
    direc = isnothing(temp) ? 0.0 : temp

    temp = HRA_indirect(h, x, y)
    indirec = isnothing(temp) ? 0.0 : temp
    return α * direc + (1 - α) * indirec
end

#####################################################



function hyperedge_dist(h::Hypergraph)
    """Returns a Distributions sampler, made after the distribution of
        hyperedge size of h
    """
    v = length.(h.he2v)
    return Spl(v)
end;


function node_dist(h::Hypergraph)
    """Returns a Distributions sampler, made from the distribution of
        node degrees of h: this supposes node weight = 1 for all nodes
    """
    return h.v2he .|> length |> Spl
end;

# function hyperedge_distrib(h::Hypergraph)
#     """Creates a histogram of the 
#     #histogram(y, normalize=:pdf, label="rand")
#     """
#     length.(h.v2he)
# end;

#################################################### 

function nodes_by_degree(h::Hypergraph)
    dd = Dict{Float64,Vector{Int64}}()
    for n in h_nodes(h)
        d = node_degree(h, n)
        if d in keys(dd)
            push!(dd[d], n)
        else
            dd[d] = [n]
        end
    end
    dd
end

#################################################### 

function nodes_degree(h::Hypergraph)
    dd = Vector{Float64}(undef, 0)
    for n = 1:nhv(h)
        deg = node_degree(h, n)
        #@show n, deg
        if !isnothing(deg)
            push!(dd, node_degree(h, n))
        end
    end
    return dd
end

####################################################

function NHAS(h::Hypergraph, x::Int, e_id::Int)

    res = 0.0
    for (y, weight) in getvertices(h, e_id) # this is a dict, we only care for the key

        temp = HRA(h, x, y)
        #@show x, y, temp, res
        if isnothing(temp)
            continue
        end
        res += temp
    end

    res /= edge_degree(h, e_id)

end

####################################################
function NHAS(h::Hypergraph, x::Int, h_edge::HyperEdge)

    if isempty(h_edge.nodes)
        return 0.0
    end
    res = 0.0
    for y in keys(h_edge.nodes)  # this is a dict, we only care for the key
        #@show y, weight, res
        temp = HRA(h, x, y)
        #@show temp
        if isnothing(temp)
            continue
        end
        res += temp
    end

    res /= length(h_edge.nodes)

end

####################################################

# function NHAS(h::Hypergraph, new_edge::HyperEdge)  x::Int, e_id::Int)

#     nhas = 0.0
#     for (y, weight) in getvertices(h, e_id) # this is a dict, we only care for the key
#         #@show y,weight, nhas
#         temp = HRA(h, x, y)
#         #@show temp
#         if isnothing(temp)
#             continue
#         end
#         nhas += temp
#     end

#     nhas /= edge_degree(h, e_id)

# end


##################################################################################
# So now we loop over all the nodes not in new_he and compute theri HNAS wrt new_he
function calc_NHAS_test(h::Hypergraph, new_he::HyperEdge)
    # Initialize
    nhas_scores = zeros(Float64, nhv(h))
    scores = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff(h_nodes(h), keys(new_he.nodes))
        sc = NHAS(h, nod, new_he)
        if sc != 0.0
            nhas_scores[nod] = sc

            for (j, weig) in new_he.nodes
                scores[nod] += HRA(h, nod, j)
            end
            scores *= 1.0 / length(new_he.e_id)
        end
    end
    return scores, nhas_scores
end

######################################################33
function calc_NHAS(h::Hypergraph, new_he::HyperEdge)
    """Node-Hyperedge Attachment Score calculated for a Hypergraph 
    and an hyperedge which is not part of it.
    """

    # we can only calculate NHAS for nodes which are NOT part of new_he
    sdif = setdiff(1:size(h)[1], keys(new_he.nodes))

    # Calculate NHAS scores
    sc = Folds.map(nod -> NHAS(h, nod, new_he), sdif)
end
#

#######################################################
function create_node_sampler(ndb::Vector{Float64})::Spl

    pooldensity = ndb |> unique |> sort

    pool = Vector{Int}(undef, 0)
    # poor man's weighted distribution.
    # oool contains node degrees. It contains n elements with value n, eg there are 5 fives and 7 sevens
    # so that when we sample it, the probability to choose degree d will be proportional to the degree.
    for i in pooldensity
        for j = 1:i
            push!(pool, i)
        end
    end

    # create the sampler
    nodeSpl = Spl(pool)

end

##################################################3
function get_random_node_by_degree(nbl, node_deg; accuracy = 0.1)
    """
        Returns a random node whose degree is within atol (see Base.isapprox) 
        of input value deg.
        if all weights == 1, then  deg should be an int, in which case ec_of_degs could be a 
        dict and the algo could be faster (no need for searches and findall etc) 
    """

    #println(">>>>  ", nbl[1:10], "  ", typeof(node_deg), " ", node_deg)
    #, findall(isapprox(deg, atol=0.1), nbl))
    res = (findall(isapprox(node_deg, atol = 0.1), nbl) |> rand)

end

##################################################
using StatsBase


function create_new_hyperedge(hg::Hypergraph; n::Int64 = 1)
    """
    Returns a vector of HyperEdges of length n, extrapolated from hg
    """
    weight = 1.0
    # Step 3 - 4: We get a hyperedge degree with the help of a sampler from the graph, before we add a new edge
    samp = hyperedge_dist(hg)
    new_hedge_size = rand(samp)
    #@show(new_hedge_size)
    #readline()
    #foreach(x -> println(x), new_hedge_sizes)

    # For a particular degree we can choose a node  
    nbd = nodes_degree(hg) #[15 11 ... 8]
    # if any(>(1e6), nbd)
    #     @show typeof(nbd), length(nbd)
    # end
    #step 5-6 get first node
    # get the distribution of node degrees
    nodeSpl = length.(hg.v2he)
    # weights cannot share memory w/ nodespl, so deepcopy
    weight_vec = (deepcopy(nodeSpl) |> fweights)

    #create_node_sampler(nbd)


    # node_degrees = rand(nodeSpl, n=n) # this is also a vector w/ 1 element, so use it as in nbd[ node_degrees[1] ]
    node_degrees = sample(nodeSpl, weight_vec, n)
    # get a random vertex from the vec of vertices w/ deg as found
    first_nodes = map(nod -> get_random_node_by_degree(nbd, nod), node_degrees)

    # the vector we will return at the end.  All hyperhegdes share the same 
    # e_id -- its just conventional
    hyperedges = Vector{HyperEdge}(undef, n)

    for n_of_edges = 1:n

        # step 1 - 2
        # So now we can add the new hyperdge, whcih contains the first node w/ weight 1.
        #add_hyperedge!(hg, vertices=Dict(first_node => 1.0))
        new_he = HyperEdge(nhe(hg) + 1, Dict([first_nodes[n_of_edges] => 1.0]))

        #@show new_hedge_size, first_node  #, length(new_he.nodes) < new_hedge_size
        # add an empty hyperedge to the (Hypergraph
        #add_hyperedge!(hg, new_he)
        # a loop over the needed number of elements in the new h-edge 
        while length(new_he.nodes) < new_hedge_size[1]

            #calculate scores and put them in a vector 
            # This is defined to be of length eq to n of nodes in the hgraph, 
            # which is a bit too much, but this way we have an easy way to know 
            # which node had which score without the need for a dict, just by the node's index.
            # And vectors are quite fast.
            scores = calc_NHAS(hg, new_he)
            if sum(scores) < eps(Float64)
                continue
            end

            # Choose a node (which is not in new_he) w/ prob proportional to its score 
            # -> the scores are weights of the sampling. so   
            # we can take only non zero scores (since 0 does not contribute anything)
            # and also we ignore the nodes in new_he (these are already in)
            indeces = findall(!=(0.0), scores)

            scores_view = getindex(scores, indeces)
            # @show indeces
            # choose a node
            newV_idx = sample(indeces, Weights(scores_view))
            #
            # if (length(new_he.nodes) % 10) == 0        
            #     @printf "length of nodes %i Chose node  %i w/ a score of %.2f \n" length(new_he.nodes) newV_idx scores[newV_idx]
            # end

            # This is added to the new hyperegde and to new_he ecdfplot
            new_he.nodes[newV_idx] = weight
            #hg[newV_idx, new_he.e_id] = 1.0

        end
        hyperedges[n_of_edges] = new_he

    end
    # we could add here one test: there shouldn't be any dublicates in new_he.nodes.
    # (or overload push! to do the checking)
    return hyperedges
end

#############################################################################
function find_disconnected_he(hyperg::Hypergraph, cv_partition)
    """
        For a given partition cv_partition = (kfold, onefold) of hypergraph hyperg, check hyperg[kfold] (ie E^T)
        for rows (ie nodes) that sum to 0. These are nodes that do not exist in hyperg[kfold] (E^M), 
        nut shoud exist in hyperg[onefold]. 
        We remove the hyperedges that contain these nodes from hyperg[onefold].
        If there are j such hyperedges, then the 'missing' set will now contain 
        onefold - j hyperedges 
    """
    (kfold, onefold) = cv_partition
    finder = find_all_empty_nodes(hyperg[:, kfold])
    isempty_finder = isempty(finder)
    @debug "is empty finder: $(Tuple(finder)) -> $(isempty_finder)"


    if isempty_finder
        # basically we do nothing: onefold is the input range cv_partition[2]
        return onefold
    end

    # ok, if we are here, finder is not empty
    discarded_he = Vector{Int64}()
    # we know that at rows of hyperg[:, kfold] = E^T given by finder 
    # there are only 0 entries. We need to find the columns in 
    # hyperg[:, onefold] = E^M that contain non zero values. 
    foreach(finder) do x
        #sanity: finder must be (n,1) a vector.
        if x[2] != 1
            throw("Error in find_disconnected_he. Exiting.")
            exit(1)
        end

        non_zero_hyperedges =
            @pipe replace(hyperg[x[1], onefold], nothing => 0) |> findall(!=(0), _)
        #println("Node $(x[1]) is zero. Checking E^M at $(non_zero_hyperedges)")
        push!(discarded_he, non_zero_hyperedges...)
        #display(hyperg[:, onefold])

    end

    # this HE set is discarded
    discarded_he |> unique! |> sort!
    println("Hyperedges #$(onefold[discarded_he]) ------   Discarded")
    onefold_copy = (onefold |> collect |> deepcopy)
    return deleteat!(onefold_copy, discarded_he)

end

#############################################################################
using ProgressMeter

function foldem(hyperg::Hypergraph, fold_k)
    """
    Perform the k-fold cross validation. 
    After partitioning the h-edges of the h-graph into k subsets, we loop over them,
    identifying them as E^T (training) and E^M (missing) sets. 
    There may exist cases where E^T does not contain some node, ie contains empty hyperedges, 
    The relevant hyperedges (including the nodes) are removed from E^M
    """


    cv = collect(kfolds(size(hyperg)[2], fold_k))

    n_loops = (cv[1] .|> length |> length)
    av_f1_scores = []
    for (k, j) in zip(cv[1], cv[2])
        # k is E^T, the training set and j the 'missing' set, E^M
        # check E^M for disconnected vertices. Return either the folded 
        # iterator if no such disconnected nodes are found, or 
        # the corrected iterator over which we are going to cross validate
        onefold = find_disconnected_he(hyperg, (k, j))
        @show onefold
        hhg = Hypergraph(hyperg[:, k])

        # Now create new h-edges for the kfold E^T hgraph, that later we will 
        # compare to onefold E^M edges. 


        new_Hedges = create_new_hyperedge(hhg, n = length(onefold))

        fs = calc_av_F1score_matrix(new_Hedges, hyperg[:, onefold])
        push!(av_f1_scores, fs)
        println("fs = $(fs)")
        println("###"^20)
    end
    return av_f1_scores
end


################################################3
function calc_av_F1score_matrix_1(Eᴾ, Eᴹ)
    """
        The 2 inputs are not of the same format. Eᴾ is a vector of hyperEdges, 
        while Eᴹ is just a view of the hypergraph, hyperg[:, onefold], the 'missing' edges 
        so some manipulation is needed.
    """
    dim = size(Eᴹ)[1]
    vsp_dim = size(Eᴾ)[1]
    # rows in hypergraph -> no of nodes, or length of the vectors
    @assert size(Eᴾ)[1] == size(Eᴹ)[2] # size of the vector space under consideration, ie no of vecs

    #@show dim, size(Eᴹ), size(Eᴾ)

    Eᴹ0 = replace(Eᴹ, nothing => 0.0)
    Eᴾ0 = map(x -> HE2Vect(x, dim), Eᴾ)
    fscore_matrix = zeros(Float64, vsp_dim, vsp_dim)
    #fscore_matrixReverse = zeros(Float64, vsp_dim, vsp_dim)

    idxlist = [(i, j) for j = 1:vsp_dim for i = 1:vsp_dim]

    foreach(idxlist) do (i, j)

        fscore_matrix[i, j] = m(Eᴹ0[:, i], Eᴾ0[:, j])

    end

    avg_F1 = 0.5 * (argmaxg_prime_sum(fscore_matrix) + argmaxg_sum(fscore_matrix))
    println("average F1 ", avg_F1)
    return avg_F1
end

##############################################################333
function argmaxg_mean(fscore_matrix)
    """ 
    the best matching missing hyperedge to each predicted hyperedge 
    !!! For different dimensions of fscore_matrix, the result of argmax_mean and 
    argmax_prime_mean are not statistically equivalent. See function test_argmax_equiv below
    """
    # first index is predicted, second missing
    # here we sum over predicted. 
    maximum(fscore_matrix, dims = 2) |> mean

end

function argmaxg_prime_mean(fscore_matrix)
    """ 
    the best-matching predicted hyperedge to each missing hyperedge
    """
    # first index is predicted, second missing
    # here we sum over missing:
    # For missing 1 (col 1, [:,1]) which row is best?
    #  this is maximum(mat, dims=1) 
    #@show size(fscore_matrix)
    maximum(fscore_matrix, dims = 1) |> mean

end

# try for n > 100000
function test_argmax_equiv(n)
    means = test_argmax_equiv_first(n)
    for (k, v) in means
        println("$k mean $(mean(v)) ") # std $(std(v))")
    end
    println(
        "The same numbers distributed in different dimension matrices do not have the same means",
    )

end

function test_argmax_equiv_first(n)

    function make_means()
        rvect = rand(1:100, 100)
        a = reshape(rvect, (2, 50))
        b = reshape(rvect, (10, 10))
        ad1 = maximum(a, dims = 1) |> mean
        ad2 = maximum(a, dims = 2) |> mean
        bd1 = maximum(b, dims = 1) |> mean
        bd2 = maximum(b, dims = 2) |> mean
        return (ad1, ad2, bd1, bd2)
    end

    means = Dict([
        "2_50_dims=1" => [],
        "2_50_dims=2" => [],
        "10_10_dims=1" => [],
        "10_10_dims=2" => [],
    ])
    foreach(1:n) do x
        x = make_means()

        push!(means["2_50_dims=1"], x[1])
        push!(means["2_50_dims=2"], x[2])
        push!(means["10_10_dims=1"], x[3])
        push!(means["10_10_dims=2"], x[4])

    end
    return means

end
