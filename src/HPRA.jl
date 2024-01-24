using SimpleHypergraphs
import Graphs
using Distributions, StatsPlots
using Printf
####################################################

# used in constructing sammples of hyperegde degrees to choose from
struct Spl <: Sampleable{Univariate,Discrete}
    vect::Vector{T} where {T<:Real}
end


####################################################
function Base.rand(spl::Spl; n=1)
    l = length(spl.vect)
    spl.vect[rand(1:l, n)]
end;

#################################################################33
# # convenince struct 
# mutable struct HyperEdge
#     e_id::Int64
#     #nodes::Vector{Union{Nothing,Int64}}
#     nodes::Dict{Int64,Float64}
# end
# HyperEdge() = HyperEdge(0, Dict{Int64,Float64}())
# HyperEdge(e::Int64) = HyperEdge(e,)
# HHyperEdge(e::Int64, v::Vector{Union{Nothing,Int64}}) = HyperEdge(e, v)
# HyperEdge(h::Hypergraph, e_idx::Int64) = HyperEdge(e_idx, h[:, e_idx])

#
#################################################################33
# convenince struct 
mutable struct HyperEdge{T}
    e_id::Int64
    nodes::Dict{T,Float64}
end

HyperEdge() = HyperEdge{Int64}(0, Dict{Int64,Float64}())
HyperEdge{T}(e::Int64) where {T} = HyperEdge(e, Dict{T,Float64}())

# This one copies a hyperedge from a hypergraph 
function HyperEdge{T}(h::Hypergraph{Float64,Nothing,Nothing,Dict{T,Float64}}, e_idx::Int64) where {T}
    hedge = HyperEdge{T}(e_idx)
    for j in eachindex(h[:, e_idx])
        if !isnothing(h[j, e_idx])
            hedge.nodes[j] = h[j, e_idx]
        end
    end
    return hedge
end
#-----------------------
import Base: ==

function ==(h1::HyperEdge, h2::HyperEdge)

    cond1 = h1.e_id == h2.e_id
    cond2 = h1.nodes == h2.nodes

    cond1 && cond2

end
#-----------------------
function h_edge_in_cont(set_of_h_edges, h_edge)
    """
    set_of_h_edges is a collection of hypegedges, h_edge is a hyperedge
    Returns a set of unique h-edges.
    """

    if any(==(h_edge), set_of_h_edges)
        return set_of_h_edges
    else
        push!(set_of_h_edges, h_edge)
    end
    return set_of_h_edges
end

#################################################################33

using Pipe: @pipe
find_empty_nodes(mat) = @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> any(==(0), _)
find_all_empty_nodes(mat) = @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> findall(==(0), _)

#################################################################33

using MLJBase, MLJ, MLUtils

struct HypergraphClassifier <: MLJBase.Probabilistic
end

# fit returns the result of applying create_new_hyperedge on input X:
function MLJBase.fit(model::HypergraphClassifier, verbosity, X, y)
    fitresult = create_new_hyperedge
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# `predict` returns the passed fitresult (pdf) for all new patterns:
MLJBase.predict(model::HypergraphClassifier, fitresult, Xnew) =
    [fitresult for r in 1:nrows(Xnew)]

#################################################################
#
function SimpleHypergraphs.add_hyperedge!(hg::Hypergraph, newhedge::HyperEdge)
    """
    An extension wnich accepts HyperEdge as argument
    For the time being only weights == 1.0 are implemented
    """
    if newhedge.e_id <= length(hg.he2v)
        #this is not an add, issue a warning
        println("Cannot add hyperedge $(newhedge.e_id), Gypergraph has $(size(hg.he2v)) h-edges. Returning")
        return false
    end

    if newhedge.e_id != size(hg)[2] + 1
        println("Cannot add hyperedge $(newhedge.e_id) to hypergraph: It contains $(size(hg)[2]) hyperdges.")
        return false
    end

    add_hyperedge!(hg, vertices=newhedge.nodes)
    return true
end

#############################################################
function node_degree(h::Hypergraph, v_id::Int; edge_size::Int=1)
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


##################################################################3
function h_Neighbours(h::Hypergraph, v_id::Int; n_commmon_edges::Int=1)
    """ a set containing the one-hop neighbors of node
    v_id (nodes of hyperedges, v is part of). The size keyword argument 
    returns the set of neighbors that share at least s edges with the given node, the default is 1.
    """
    eds = gethyperedges(h, v_id)
    neighb = Set()
    for (e, w) in eds
        vs = getvertices(h, e)
        #println(vs)
        if length(vs) >= n_commmon_edges
            for (v, ww) in vs
                if v == v_id
                    continue
                end
                push!(neighb, v)
            end
        end
    end
    return neighb
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
function h_nodes(h::Hypergraph)  #; number_format::Bool = true, ttor::Dict = ttor)
    # size(h) returns (n_vertices, n_edges)
    # h.v2he is a vector of the hyperedges ie , it does what the following loop woudl do: 
    # for ed_n in 1:size(h)[1] #gethyperedges(h,v_id)
    #     # gethyperedges(h,ed_n) returns the dict of the edges 
    #     # that contain vert er_n with the respecthve weights
    #     eds =  gethyperedges(h,ed_n)
    #     @show  eds  
    # end
    res_vert = []
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
function RA(h::Hypergraph, x::Int, y::Int; n_commmon_edges::Int=1, edge_size::Int=1)
    L"""
 #   Resource Allocation  of 2 *not directly connected* nodes x, y is   
    the \sum_{z∈ [N(x) ∩ N(y)]} \frac{1}{d(z)},
    where N(x) the neighbours of x, and d(x) the degree of x.
    """

    nx = h_Neighbours(h, x) #, n_commmon_edges=n_commmon_edges)
    if y in nx
        return nothing
    end

    ny = h_Neighbours(h, y, n_commmon_edges=n_commmon_edges)
    ra = 0.0
    for z in nx ∩ ny
        ra += 1 / node_degree(h, z, edge_size=edge_size)
        #println("---- ",z, "  ",node_degree(h, z, edge_size= edge_size) )
    end
    return ra
end


function HRA_direct(h::Hypergraph, x::Int, y::Int; n_commmon_edges::Int=1, edge_size::Int=1)
    L"""
    Direct part of Hyper Resource Allocation.   
    HRA_{direct} (x, y) = \sum_{e, s.t. x,y ∈e} 1/ [δ(e) - 1] 
    (is this equal to A_ndp?) 
    """
    if x == y
        return nothing
    end
    # first we need the hedges than include both x an y
    edsx = (gethyperedges(h, x) |> keys)
    edsy = (gethyperedges(h, y) |> keys)

    common_edges = intersect(edsx, edsy)
    if common_edges == Set()
        return nothing
    end

    hrad = 0.0
    for ed in common_edges
        hr = edge_degree(h, ed) - 1
        hrad += 1 / hr

    end
    return hrad
end

function HRA_indirect(h::Hypergraph, x::Int, y::Int; n_commmon_edges::Int=1, edge_size::Int=1)
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

function HRA(h::Hypergraph, x::Int, y::Int)
    if x == y
        return nothing
    end
    temp = HRA_direct(h, x, y)
    direc = isnothing(temp) ? 0.0 : temp

    temp = HRA_indirect(h, x, y)
    indirec = isnothing(temp) ? 0.0 : temp
    return direc + indirec
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
    for n in 1:nhv(h)
        deg = node_degree(h, n)
        #@show n, deg
        if isnothing(deg)
            continue
        end
        push!(dd, node_degree(h, n))
    end
    return dd
end

####################################################

function NHAS(h::Hypergraph, x::Int, e_id::Int)

    nhas = 0.0
    for (y, weight) in getvertices(h, e_id) # this is a dict, we only care for the key
        #@show y,weight, nhas
        temp = HRA(h, x, y)
        #@show temp
        if isnothing(temp)
            continue
        end
        nhas += temp
    end

    nhas /= edge_degree(h, e_id)

end

####################################################
function NHAS(h::Hypergraph, x::Int, h_edge::HyperEdge)

    if isempty(h_edge.nodes)
        return 0.0
    end
    nhas = 0.0
    for (y, weight) in h_edge.nodes # this is a dict, we only care for the key
        #@show y,weight, nhas
        temp = HRA(h, x, y)
        #@show temp
        if isnothing(temp)
            continue
        end
        nhas += temp
    end

    nhas /= length(h_edge.nodes)

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

function calc_NHAS(h::Hypergraph, new_he::HyperEdge)
    # Initialize

    scores = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff(h_nodes(h), keys(new_he.nodes))
        sc = NHAS(h, nod, new_he)
        if sc != 0.0
            scores[nod] = sc

            # for j in new_he.nodes
            #     scores[nod] +=  HRA(h, nod, j)
            # end
            # scores *= 1. / length(new_he.e_id)
        end
    end
    return scores
end
# ####################################################
# # find the disrtibution of the NHAS scores for existing nodes and put them in a dict
# function get_node_distr_probs(h::Hypergraph)
#     node_prob_distr = Dict{Int64,Float64}()
#     for inode in h_nodes(h)
#         #push!(node_prob_distr, node_degree(h, inode))
#         node_prob_distr[inode] = node_degree(h, inode)
#     end
#     return node_prob_distr
# end

function create_node_sampler(ndb::Vector{Float64})::Spl

    pooldensity = ndb |> unique |> sort

    pool = Vector{Int}(undef, 0)
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

end

##################################################3
function get_random_node_by_degree(nbl, node_deg; accuracy=0.1)
    """
        Returns a random node whose degree is within atol (see Base.isapprox) 
        of input value deg.
        if all weights == 1, then  deg should be an int, in which case ec_of_degs could be a 
        dict and the algo could be faster (no need for searches and findall etc) 
    """

    #println(">>>>  ", nbl[1:10], "  ", typeof(node_deg), " ", node_deg)
    #, findall(isapprox(deg, atol=0.1), nbl))
    res = (findall(isapprox(node_deg[1], atol=0.1), nbl) |> rand)

end