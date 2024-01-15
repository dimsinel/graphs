using SimpleHypergraphs
import Graphs
using Distributions, StatsPlots
####################################################

# used in constructing sammples of hyperegde degrees to choose from
struct Spl <: Sampleable{Univariate,Discrete}
    vect::Vector{T} where {T<:Real}
end

# #used in constructing sammples of nodes
# mutable struct SplInt <: Sampleable{Univariate,Discrete}
#     vect::Vector{Int64}
# end

####################################################
function Base.rand(spl::Spl;n=1)
    l = length(spl.vect)
    spl.vect[rand(1:l,n)]
end;

#################################################################33
# convenince struct 
mutable struct NewHEdge
    e_id::Int64
    nodes::Vector{Union{Nothing,Int64}}
end
NewHEdge() = NewHEdge(0, [])
NewHEdge(e::Int64) = NewHEdge(e, [])


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
    if degr > 0.0
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

function nodes_degree(h::Hypergraph)::Vector{Float64}
    dd = Vector{Float64}(undef, nhv(Hcitecoref))
    for n in 1:nhv(Hcitecoref)
        #@show n, node_degree(h, n)
        dd[n] = node_degree(h, n)
    end
    return dd
end

##################################################################################
# So now we loop over all the nodes not in new_he and compute theri HNAS wrt new_he
function calc_NHAS_test(h::Hypergraph, new_he::NewHEdge)
    # Initialize
    nhas_scores = zeros(Float64, nhv(h))
    scores = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff(h_nodes(h), new_he.nodes)
        sc = NHAS(h, nod, new_he.e_id)
        if sc != 0.0
            nhas_scores[nod] = sc

            for j in new_he.nodes
                scores[nod] += HRA(h, nod, j)
            end
            scores *= 1.0 / length(new_he.e_id)
        end
    end
    return scores, nhas_scores
end

function calc_NHAS(h::Hypergraph, new_he::NewHEdge)
    # Initialize

    scores = zeros(Float64, nhv(h))

    # Calculate NHAS scores in 2 ways as a test.
    for nod in setdiff(h_nodes(h), new_he.nodes)
        sc = NHAS(h, nod, new_he.e_id)
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