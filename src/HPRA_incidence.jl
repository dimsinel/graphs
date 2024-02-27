using SimpleHypergraphs
import Graphs
using LinearAlgebra
using SparseArrays
using SuiteSparseGraphBLAS


# convenince struct 
abstract type myAbstractHyperGraph{T<:Number,U<:Number} end

struct myHyperGraph{T<:Integer,U<:Real} <: myAbstractHyperGraph{T,U} # T=Int, U =Float
    e_id::T                          # n of hyperedges (ie columns)
    v_id::T                          # n of nodes (rows)
    H::Matrix{U}                     # Incidence Matrix
    nodes::Diagonal{T}               # D_v ∈ R^{v_id × v_id} the diagonal matrix containing node degrees, 
    h_edges::Diagonal{T}             # D_e ∈ R^{e_id × e_id}     diagonal matrix of hyperedge degrees 
    weights::Diagonal{U}          # D_w ∈ R^{e_id × e_id}     diagonal matrix of hyperedge weights
    v_neighbours::Dict{T,Set{T}} # the set of neighbours for each node
    #Andp::Matrix{Float64}
end

##########################################################

# --- struct SparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
struct mySparseHyperGraph{T<:Integer,U<:Real} <: myAbstractHyperGraph{T,U} # T=Int, U =Float
    e_id::T                          # n of hyperedges (ie columns)
    v_id::T                          # n of nodes (rows)
    #H::SparseMatrixCSC{U}               # Incidence Matrix
    H::GBMatrix{U}               # Incidence Matrix
    nodes::GBMatrix{T}               # D_v ∈ R^{v_id × v_id} the diagonal matrix containing node degrees, 
    h_edges::GBMatrix{T}             # D_e ∈ R^{e_id × e_id}     diagonal matrix of hyperedge degrees 
    weights::GBMatrix{U}          # D_w ∈ R^{e_id × e_id}     diagonal matrix of hyperedge weights
    v_neighbours::Dict{T,Set{T}} # the set of neighbours for each node
    #Andp::Matrix{Float64}
end

######################################################################
myHyperGraph{T,U}(e::T, v::T) where {T<:Number,U<:Number} = myHyperGraph(
    e,
    v,
    [1.0 0.0; 0.0 1.0],
    Diagonal([1, 1]),
    Diagonal([1, 1]),
    Diagonal([1.0, 1.0]),
    Dict{T,Set{T}}())


# function myHyperGraph{T}(e::T, v::T) where {T<:Number, U <: Number}
#     myHyperGraph(e::T, v::T,
#         Matrix{T}(undef,2,2),
#         Diagonal(ones(T, 2)),
#         Diagonal(ones(U, 2)),
#         Dict{T,Set{T}}())
#         #zeros(U, 1, 1)) # won't be used in this cstor, so leave it small
# end 

function myHyperGraph{T,U}(D::Diagonal{T}, E::Diagonal{T}) where {T<:Number,U<:Number}
    myHyperGraph(
        size(E, 1),
        size(D, 1),
        zeros(U, size(D, 1), size(E, 1)),
        D,
        E,
        Diagonal(ones(U, size(E, 1))),
        Dict{T,Set{T}}())
    #zeros(Float64, 1, 1))
end

function myHyperGraph(D::Vector{Int64}, E::Vector{Int64})
    myHyperGraph(Diagonal(D), Diagonal(E))
end
# --------------------------------------------------------------------------------
function myHyperGraph(h::T) where {T<:Union{Hypergraph,Matrix{Union{Nothing,Real}}}}

    H = Incidence(h)
    D = nodes_degree_mat(h)
    E = hyperedges_degree_mat(h)
    eid = size(E, 1)
    nid = size(D, 1)
    W = hyper_weights(h) |> Diagonal

    d = Dict{Int64,Set{Int64}}()
    for i = 1:nid
        d[i] = h_Neighbours(h, i)
    end

    @info "[myHyperGraph cstor] Hypergraph edges=$(eid), nodes=$(nid)"

    # Ei = E - I
    # DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)] |> Diagonal)
    # A_ndp = H * W * DeInv * H' - D
    return myHyperGraph{Int64,Float64}(eid, nid, H, D, E, W, d)

end

#--------------------------------------------------------

function mySparseHyperGraph(h::T) where {T<:Union{Hypergraph,AbstractSparseMatrix,Matrix{Union{Nothing,Real}}}}

    H = Incidence(h) |> GBMatrix
    D = nodes_degree_mat(h) |> GBMatrix
    E = hyperedges_degree_mat(h) |> GBMatrix
    eid = size(E, 1)
    nid = size(D, 1)
    W = GBMatrix([1:size(h, 2)...], [1:size(h, 2)...], hyper_weights(h))
    d = Dict{Int64,Set{Int64}}()
    for i = 1:nid
        d[i] = h_Neighbours(h, i)
    end

    @debug "[myHyperGraph cstor] Hypergraph edges=$(eid), nodes=$(nid)"

    # Ei = E - I
    # DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)] |> Diagonal)
    # A_ndp = H * W * DeInv * H' - D
    return mySparseHyperGraph{Int64,Float64}(eid, nid, H, D, E, W, d)

end
##############################################
function SimpleHypergraphs.gethyperedges(h::T, v_id) where {T<:AbstractArray}

    d = Dict{Int64,Bool}()
    for i in eachindex(h[v_id, :])
        if h[v_id, i] != 0
            d[i] = Bool(h[v_id, i])
        end
        @debug "i=$i,val=$(val)"
    end
    return d
end
##############################################
function SimpleHypergraphs.getvertices(h::T, e_id) where {T<:AbstractMatrix}

    d = Dict{Int64,Bool}()
    for i in eachindex(h[:, e_id])
        if h[i, e_id] != 0
            d[i] = Bool(h[i, e_id])
        end
        @debug "i=$i,val=$(val)"
    end
    return d
end
##############################################

struct mySubHyperGraph{T,U} <: myAbstractHyperGraph{T,U}

    e_id::Any                        # n of hyperedges (ie columns)
    v_id::Any                         # n of nodes (rows)
    H::SubArray                             # Incidence Matrix
    nodes::Diagonal              # D_v ∈ R^{v_id × v_id}     diagonal matrix containing node degrees, 
    h_edges::Diagonal             # D_e ∈ R^{e_id × e_id}     diagonal matrix of hyperedge degrees 
    weights::Diagonal             # D_w ∈ R^{e_id × e_id}     diagonal matrix of hyperedge weights
    v_neighbours::Dict            # the set of neighbours for each node
    #Andp:::SubArray  

end

function mySubHyperGraph(h::myHyperGraph, k::T) where {T<:Vector}
    """
            The input h is a myHyperGraph which will be partitioned, kfold style, by k
            This is 3-4 times slower than a myHyperGraph, which copies h.H[:, k] to 
            an intermediate matrix instead of creating a view
    """
    H = view(h.H, :, k)

    D = nodes_degree_mat(H)
    E = hyperedges_degree_mat(H)

    eid = size(E, 1)
    nid = size(D, 1)

    # sanity 
    @assert nid == h.v_id # must remain the same
    @debug "[mySubHyperGraph cstor] eid=$eid  h.e_id=$(h.e_id) length(k)=$(length(k)) size(H)=$(size(H))"
    @assert eid == length(k)
    W = hyper_weights(H) |> Diagonal

    d = Dict{Int64,Set{Int64}}()
    for i = 1:nid
        d[i] = h_Neighbours(H, i)
    end

    mySubHyperGraph{typeof(nid),eltype(H)}(eid, nid, H, D, E, W, d)

end


## ====================================== ##
import Base: ==
function ==(h1::myAbstractHyperGraph, h2::myAbstractHyperGraph)

    eq1 = h1.e_id == h2.e_id
    eq2 = h1.v_id == h2.v_id
    eq3 = h1.H == h2.H
    @debug eq1, eq2, eq3

    eq3 = h1.nodes == h1.nodes
    eq4 = h1.h_edges == h2.h_edges
    eq5 = h1.weights == h1.weights

    dcond1 = h1.v_neighbours == h2.v_neighbours
    return eq1 && eq2 && eq3 & eq4 && eq5 && dcond1

end
#-----------------------------------------------------------

Base.hash(x::myAbstractHyperGraph, h::UInt) = hash(:myAbstractHyperGraph, hash(x.nodes, h))

###########################################

# function compare_dicts(d1, d2)
#     lastcond = true
#     for k in keys(d1)
#         @debug k, d1[k]
#         @debug d2[k]
#         if d1[k] != d2[k]
#             lastcond = false
#             @debug "break"
#             break
#         else
#             continue
#         end
#     end
#     return lastcond
# end

##############################################

sgnode_density(hh::T) where {T<:myAbstractHyperGraph} = sum(hh.H) / (hh.e_id * hh.v_id)
node_density(h::Hypergraph) = sum(Incidence(h)) / (length(h.he2v) * length(h.v2he))

##############################################

##################################################################3
function h_Neighbours(h::T, v_id::Int; n_commmon_edges::Int=1) where {T} ## this is faster than v_neigh
    """Reteurns a set containing the one-hop neighbors of node
    v_id (nodes of hyperedges, v is part of). The n_commmon_edges keyword argument 
    returns the set of neighbors that share at least s edges with the given node.
    As is, works for both myAbstractHyperGraphs AND Hypergrapgs
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
######################################################
function nodes_degree_mat(h::Hypergraph)
    """Diagonal matrix of node degrees. Practically the same as node_dist"""

    # This...
    # D = Base.mapreduce(x -> (x != (0) && !isnothing(x)), +, h, dims=2)
    # Diagonal([D...])
    # and this...
    D = Diagonal(length.(h.v2he))
    # ...are the same

end

function nodes_degree_mat(H::T) where {T<:AbstractMatrix}
    """Diagonal matrix of node degrees. Practically the same as node_dist"""

    # This...
    D = Base.mapreduce(x -> (x != (0) && !isnothing(x)), +, H, dims=2)
    Diagonal([D...])
    # and this...
    #D = Diagonal(length.(h.v2he))
    # ...are the same

end
##############################################
function hyperedges_degree_mat(h::Hypergraph)
    """
    Diagonal matrix of hyperedge degrees. Practically the same as hyperedge_dist
    """
    #  # This
    # D = mapreduce(x -> x != (0) && !isnothing(x), +, h, dims=1)
    # Diagonal([D...])
    #...and this
    D = Diagonal(length.(h.he2v))
    # ..are the same
end

function hyperedges_degree_mat(H::T) where {T<:AbstractMatrix}
    """
    Diagonal matrix of hyperedge degrees. Practically the same as hyperedge_dist
    """
    #  # This
    D = mapreduce(x -> x != (0) && !isnothing(x), +, H, dims=1)
    Diagonal([D...])
    #...and this
    #D = Diagonal(length.(h.he2v))
    # ..are the same
end

###############################################
function hyper_weights(h)
    """
    Diagonal matrix of weights. For teh time being, just set them all equal to 1
    """
    ones(size(h, 2))
end

###############################################
function Incidence(h)

    return (replace(h, nothing => 0.0, true => 1.0))

end

#const H(h) = Incidence(h)
###############################################


A(h::myHyperGraph) = h.H * h.weights * h.H' - h.nodes
# adjacency, SimpleHypergraphs already had one.
# Anyway, it's a cross check.

###############################################

function Andp(h::myHyperGraph)

    Ei = h.h_edges - I
    DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)] |> Diagonal)
    return h.H * h.weights * DeInv * h.H' - h.nodes
end

#------------------------------------------------------

function Andp(h::mySparseHyperGraph)

    Ei = h.h_edges - I

    DeInv_vec = [Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)]
    DeInv = GBMatrix([eachindex(DeInv_vec)...], [eachindex(DeInv_vec)...], DeInv_vec)
    return h.H * h.weights * DeInv * h.H' - h.nodes
end

#------------------------------------------------------

function Andp(h::Hypergraph)

    H = Incidence(h)
    D = nodes_degree_mat(h)
    E = hyperedges_degree_mat(h)
    eid = size(E, 1)
    nid = size(D, 1)
    W = hyper_weights(h) |> Diagonal

    # d = Dict{Int64,Set{Int64}}()
    # for i = 1:nid
    #     d[i] = h_Neighbours(h, i)
    # end
    Ei = E - I
    DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in axes(Ei, 1)] |> Diagonal)
    Andp = H * W * DeInv * H' - D
end

####################################################
function choose_new_vertex(hg, andp, new_he)::Int64
    #calculate scores and put them in a vector 
    # This is defined to be of length eq to n of nodes in the hgraph, 
    # which is a bit too much, but this way we have an easy way to know 
    # which node had which score without the need for a dict, just by the node's index.
    # And vectors are quite fast.
    scores = calc_all_NHAS(hg, andp, new_he)
    if sum(scores) < eps(Float64)
        return 0
    end

    # Choose a node (which is not in new_he) w/ prob proportional to its score 
    # -> the scores are weights of the sampling. so   
    # we can take only non zero scores (since 0 does not contribute anything)


    indeces = Folds.findall(!=(0.0), scores)
    scores_view = getindex(scores, indeces)

    # and also we ignore the nodes in new_he (these are already in) 
    # so we set their sampling weight to 0
    vertices2exclude = findall(indeces) do x
        x ∈ keys(new_he.nodes)
    end
    #@show vertices2exclude
    scores_view[vertices2exclude] .= 0.0

    if sum(scores_view) ≈ 0
        #println("No more vertices available! Return nothing")
        return 0
    end

    # we cant have the current nodes in the sample of indeced
    #nodes2rm = findall(keys(new_he.nodes), indeces)


    # choose a node
    return sample(indeces, Weights(scores_view))
    #@show newV_idx, indeces, scores_view
    #
    # if (length(new_he.nodes) % 10) == 0        
    #     @printf "length of nodes %i Chose node  %i w/ a score of %.2f \n" length(new_he.nodes) newV_idx scores[newV_idx]
    # end


    # This is added to the new hyperegde and to new_he ecdfplot 
    #new_he.nodes[newV_idx] = 1 # this may be a name or smtng else i dont know what
    #hg[newV_idx, new_he.e_id] = 1.0
    #return new_he, true
end
# #############################################################
# function create_new_hyperedge(hg::myHyperGraph; n::Int64=1)

#     # choose the degrees of the new hyperedges according to current h_edge degrees
#     new_hedges_sizes = sample(diag(hg.h_edges), n)
#     #@show new_hedges_sizes
#     # choose new first nodes according to current nodes degree  using Preferential Attachment
#     # we dont care about the current no of nodes, only their degrees, (should we use 'unique' here?)
#     nodes_distribution = (hg.nodes |> diag |> fweights)
#     first_nodes = sample(1:hg.v_id, nodes_distribution, n)
#     #@show any(==(0), first_nodes) 
#     #sanity
#     #do this for n > 1000 to chech if node sampling was OK: d should be approx nodes_distribution
#     #= begin) # |> unique)
#     # we will sample according to this distribution, 
#     nodes_distribution = (nodes_distribution |> fw
#         ufn = unique(first_nodes)
#         fnode_distr = map(i -> count(==(i), first_nodes), ufn)
#         dd = Dict(zip(ufn, fnode_distr)) |> sort
#         norm = dd[1]
#         d = Vector{Float64}(undef, length(dd))
#         for (i, k) in dd
#             d[i] = dd[i] / norm * nodes_distribution[1]
#         end
#         @show unique(first_nodes), fnode_distr, nodes_distribution
#         @show d
#     end =#

#     #create a vector to hold the output
#     #hyperedges = Vector{HyperEdge}(undef, n)
#     hyperedges = Vector{HyperEdge}()

#     # first_noes and new_hedges_sizes are vectors of the same lenght
#     #for i in eachindex(first_nodes)
#     eternal_loop_break = 1
#     ii = 1
#     while length(hyperedges) < n
#         # There are cases when no new hyperedges can be found, 
#         # ie when n is too large relative to v_id and it exhaists all possible cases.
#         # this breaks the loop 
#         # must think of a combinatorial argument that calculates a priori 
#         # the nuymber of possible new hyperedges instead of this simple hack
#         if eternal_loop_break > n
#             break
#         end
#         ii += 1
#         i = ii % 10 + 1
#         # well, all of them are going to have the same hyperedge number, but it is immaterial here.
#         new_he = HyperEdge(hg.e_id + 1, Dict([first_nodes[i] => 1]), 1.0)

#         # Now we must add to each hyperedge[i], new_hedge_size[i]-1 nodes.
#         # a loop over the needed number of elements in the new h-edge 
#         while length(new_he.nodes) < new_hedges_sizes[i]

#             new_he, check = choose_new_vertex(hg, new_he)
#             if !check
#                 #@show "not check", i, check, length(new_he.nodes), new_hedges_sizes[i]
#                 eternal_loop_break += 1
#                 break
#             end
#         end
#         #@show "new_he", new_he
#         if new_he ∉ hyperedges
#             push!(hyperedges, new_he)
#         else
#             continue
#         end


#     end


#     #@show length(new_hedges_sizes), length(hyperedges)
#     return hyperedges
# end


#############################################################

function create_new_hyperedge(hg::T, andp::M; n::Int64=1) where {T<:myAbstractHyperGraph,M<:AbstractMatrix}


    #create a vector to hold the output
    #hyperedges = Vector{HyperEdge}(undef, n)
    hyperedges = Vector{HyperEdge}()

    new_hedges_sizes(i) = sample(diag(hg.h_edges), i)

    function first_nodes(i)
        nodes_distribution = (hg.nodes |> diag |> fweights)
        return sample(1:hg.v_id, nodes_distribution, i)
    end

    while length(hyperedges) < n
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

            new_Vertex = choose_new_vertex(hg, andp, new_he)
            if new_Vertex == 0
                # something went wrong in the new vertex calculation. 
                # Start again 
                break
            end
            new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what
            #keys(new_he.nodes)
        end
        if new_Vertex == 0 # go back to the Start
            continue
        end

        # add the new vertex
        new_he.nodes[new_Vertex] = 1 # this may be a name or smtng else i dont know what

        # double check: 
        if new_he ∉ hyperedges
            push!(hyperedges, new_he)
        else
            # ... again start this hyperedge from the start 
            continue
        end

    end
    #@show length(new_hedge_size), length(hyperedges)
    return hyperedges
end

#########################################################################################
HRA_direct(h::H) where {H<:myAbstractHyperGraph} = Andp(h)
###--------------------------------------------------------------------------------------
function HRA_indirect(h::H, andp::M, x::Int, y::Int) where {H<:myAbstractHyperGraph,M<:AbstractMatrix} # n_commmon_edges::Int=1, edge_size::Int=1)
    L"""
        Inirect part of Hyper Resource Allocation.   
        HRA_{indirect} (x, y) =  \sum_{z ∈N (x) ∩ N(y)} 
        HRA_{direct} (x,z) × \frac{1}{d(z)}  × HRA_{direct} (z, y) = A_ndp
    """

    # The intersection of neighbours below, by construction does not contain x or y, unless x = y
    if x == y
        return 0.0
    end
    # first we need the neighborhoods of x and y
    #@show x, y
    common_nodes = h.v_neighbours[x] ∩ h.v_neighbours[y]

    hra_ind = 0.0
    for z in common_nodes
        if h.nodes[z, z] == 0
            continue
        end
        # Andp is HRA_direct as we know
        temp = andp[x, z] / h.nodes[z, z]
        temp *= andp[z, y]

        hra_ind += temp

    end

    return hra_ind
end
##-------------------------------------------------------------

HRA(h::H, andp, i::Int, j::Int; α=0.5) where {H<:myAbstractHyperGraph} =
    α * andp[i, j] + (1 - α)HRA_indirect(h, andp, i, j)

################################################

function calc_all_HRA_pairs(h::H) where {H<:myAbstractHyperGraph}
    """Node-Hyperedge Attachment Score calculated for a Hypergraph 
    and an hyperedge which is not part of it.
    """
    andp = Andp(h)
    HRA_Pairs = Matrix{Float64}(undef, h.v_id, h.v_id)
    for nod1 in 1:h.v_id
        for nod2 in (nod1+1):h.v_id
            HRA_Pairs[nod1, nod2] = HRA(h, andp, nod1, nod2)
        end
        HRA_Pairs[nod1, nod1] = 0.0
        if nod1 % 100 == 1
            @info "calc_all_HRA_pairs nod1=$(nod1)"
        end
    end
    return Symmetric(HRA_Pairs)
end

####################################################
####################################################

function NHAS(h::H, andp, vertex::Int, e_id::Int) where {H<:myAbstractHyperGraph}
    """Node-Hyperedge Attachment Score.
    """

    res = 0.0
    for i in setdiff(findall(==(1), h.H[:, e_id]), [vertex])

        res += HRA(h, andp, vertex, i)
        @info i, HRA(h, andp, vertex, i), res
    end
    res /= h.h_edges[e_id]
end

# much slower
# @time begin
#     res2 = mapreduce(i -> HRA(HG, vertex, i), +, setdiff(findall(==(1), HG.H[:, j]), [vertex]))
#     res2 /= HG.h_edges[j]
# end

####################################################

function NHAS(h::H, andp, vertex::Int, h_edge::HyperEdge) where {H<:myAbstractHyperGraph}
    """Node-Hyperedge Attachment Score.
    """

    res = 0.0

    for i in setdiff(keys(h_edge.nodes), [vertex])

        res += HRA(h, andp, vertex, i)
        @debug "function NHAS", i, HRA(h, andp, vertex, i), res
    end
    res /= length(h_edge.nodes)
end


#

####################################################

function calc_all_NHAS(h::H, andp, new_he::HyperEdge{T}) where {H<:myAbstractHyperGraph,T<:Number}
    """Node-Hyperedge Attachment Score calculated for a Hypergraph 
    and an hyperedge which is not part of it.
    """

    # we can only calculate NHAS for nodes which are NOT part of new_he
    sdif = setdiff(1:h.v_id, keys(new_he.nodes))
    @debug "calc_all_NHAS: setdiff(1:h.v_id, keys(new_he.nodes)) = $(length(sdif))"
    # Calculate NHAS scores
    tempvec = Float64[]
    for nod in sdif
        nhas = NHAS(h, andp, nod, new_he)
        push!(tempvec, nhas)
        if nod % 200 == 1
            @debug "calc_all_NHAS: nod $(nod)"
        end
    end
    return tempvec
    #sc = map(nod -> NHAS(h, nod, new_he), sdif)
    #sc = Folds.map(nod -> NHAS(h, nod, new_he), sdif)
end
#################################################################33

using Pipe: @pipe
find_empty_nodes(mat) =
    @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> any(==(0), _)
find_all_empty_nodes(mat) =
    @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> findall(==(0), _)

#############################################################################

function find_connected_he(hyperg::myAbstractHyperGraph, cv_partition)
    """
        For a given partition cv_partition = (kfold, onefold) of myHyperGraph hyperg, check hyperg.H[kfold] (ie E^T)
        for rows (ie nodes) that sum to 0. These are nodes that do not exist in E^M, 
        but should exist in hyperg[onefold]. 
        We remove the hyperedges that contain these nodes from hyperg[onefold].
        If there are j such hyperedges, then the 'missing' set will now contain 
        onefold - j hyperedges. 
    """
    (kfold, onefold) = cv_partition
    finder = find_all_empty_nodes(hyperg.H[:, kfold])
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
            throw("Error in find_connected_he. Exiting.")
            exit(1)
        end

        non_zero_hyperedges =
            @pipe replace(hyperg.H[x[1], onefold], nothing => 0) |> findall(!=(0), _)
        #println("Node $(x[1]) is zero in E^T. Checking E^M at $(non_zero_hyperedges)")
        push!(discarded_he, non_zero_hyperedges...)
        #display(hyperg[:, onefold])

    end

    # this HE set is discarded
    #discarded_he |> unique! |> sort!
    kept_he = setdiff(1:length(onefold), discarded_he)

    @debug "Hyperedges #$(onefold[discarded_he]) discarded ------   we keep $(onefold[kept_he])"
    #onefold_copy = (onefold |> collect |> deepcopy)
    return onefold[kept_he]

end
################################################3

function create_mat(new_Hedges)
    """Takes a vector of n hyperedges, each containing  k nodes,
     and returns a k × n matrix
     """
    if length(new_Hedges) == 0
        return []
    end
    matdim = new_Hedges[begin].v_size
    map(x -> HE2Vect(x, matdim), new_Hedges) |> stack
end

################################################3
#function calc_av_F1score_matrix(Eᴾ::Array{Union{Nothing,T},2}, Eᴹ::Array{Union{Nothing,U},2}) where {T<:Real,U<:Real}
function calc_av_F1score_matrix(Eᴾ::T, Eᴹ::U) where {T,U<:AbstractMatrix}
    """
    The 2 inputs are nodes × h_edges matrices. 
    Eᴾ is a vector of hyperEdges produced by simulation, 
    while Eᴹ is a view of the hypergraph, hyperg[:, onefold], the 'missing' edges,  
    where onefold = k:(k+h_dim) for some k.
"""
    try
        @assert size(Eᴹ)[1] == size(Eᴾ)[1]
    catch AssertionError
        @error "size(Eᴹ)=$(size(Eᴹ)), size(Eᴾ)=$(size(Eᴾ)). Not the same no of nodes. Exiting"
        exit()
    end


    Eᴹ0 = replace(Eᴹ, nothing => 0.0)
    m_dim = size(Eᴹ)[2]
    p_dim = size(Eᴾ)[2]
    fscore_matrix = zeros(Float64, m_dim, p_dim)
    #fscore_matrixReverse = zeros(Float64, v_dim, v_dim)

    for j = 1:p_dim
        for i = 1:m_dim

            fscore_matrix[i, j] = m(Eᴹ0[:, i], Eᴾ[:, j])
            #fscore_matrixReverse[i, j] = m(Eᴾ[:, i], Eᴹ0[:, j])
            fscore_matrix[i, j] == 0.0 ? nothing :
            @info "fscore_matrix[$i, $j] = $(fscore_matrix[i, j])"
            (sum(Eᴹ0[:, i]) != 0.0 && sum(Eᴾ[:, j]) != 0.0) ? nothing :
            @info "[$i, $j], sum(Eᴹ0[:, i])= $(sum(Eᴹ0[:, i]))  sum(Eᴾ[:, j])=$(sum(Eᴾ[:, j]))"
        end
    end
    #display(fscore_matrix)), 
    avg_F1 = 0.5 * (argmaxg_prime_mean(fscore_matrix) + argmaxg_mean(fscore_matrix))
    println("average F1 ", avg_F1)
    return avg_F1 #, fscore_matrix #, fscore_matrixReverse
end

#############################################################################

function foldem(hyperg::myHyperGraph, fold_k; use_view_matrix=false)
    """
    Perform the k-fold cross validation. 
    After partitioning the h-edges of the h-graph into k subsets, we loop over them,
    identifying them as E^T (training) and E^M (missing) sets. 
    There may exist cases where E^T does not contain some node, ie contains empty hyperedges, 
    The relevant hyperedges (including the nodes) are removed from E^M
    """
    cv = collect(kfolds(hyperg.e_id, fold_k)) # breaks  1:e_id in fold_k grpups for cross validation

    #fscore_matrix, fscore_matrixReverse = Float64[], Float64[]
    #n_loops = (cv[1] .|> length |> length)
    av_f1_scores = Float64[]
    for (k, j) in zip(cv[1], cv[2])
        # k is E^T, the training set and j the 'missing' set, E^M
        # check E^M for disconnected vertices. Return either the folded 
        # iterator if no such disconnected nodes are found, or 
        # the corrected iterator over which we are going to cross validate
        kept_hedges = find_connected_he(hyperg, (k, j))
        if kept_hedges == []
            continue # no hyperedges can be predicted in this fold
        end
        @info onefold #,kept_hedges
        hhg = nothing
        if use_view_matrix
            hhg = mySubHyperGraph(hyperg, k) #, collect(onefold))
        else
            hhg = replace(hyperg.H[:, k], 0 => nothing) |> Hypergraph |> myHyperGraph
        end
        # Now create new h-edges for the kfold E^T hgraph, that later we will 
        # compare to onefold E^M edges. 

        andp = Andp(hhg)
        #new_Hedges = create_new_hyperedge(hhg, n=length(kept_hedges))
        new_Hedges = create_new_hyperedge(hhg, andp, n=length(j))

        new_Hedges_mat = create_mat(new_Hedges)
        @info typeof(hyperg.H[:, kept_hedges]), typeof(new_Hedges_mat)
        fs = calc_av_F1score_matrix(new_Hedges_mat, hyperg.H[:, kept_hedges])
        push!(av_f1_scores, fs)
        println("fs = $(fs)")
        println("###"^20)
        #print(stdout, ">>>> ")
        #read(stdin, 1)

    end
    return av_f1_scores #, fscore_matrix, fscore_matrixReverse
end
