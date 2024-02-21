using SimpleHypergraphs
import Graphs
using LinearAlgebra
using SparseArrays
using StaticArrays
using SuiteSparseGraphBLAS
#using Tullio

#mult(M, Q) = @tullio P[x, y] := M[x, c] * Q[y, c]  # sum over c ∈ 1:7 -- matrix multiplication

# a parameter needed for tuning reasons
const global size_of_change_2_sparse = 1_000_000

#########################################################

abstract type myAbsHyperGraph{T<:AbstractMatrix,U<:AbstractMatrix} end

# convenince struct 
struct myHyperGraph{T<:AbstractMatrix,U<:AbstractMatrix} <: myAbsHyperGraph{T,U}
    e_id::Int32                          # n of hyperedges (ie columns)
    v_id::Int32                          # n of nodes (rows)g
    H::T                                 # Incidence Matrix
    nodes::Diagonal               # D_v ∈ R^{v_id × v_id}     diagonal matrix containing node degrees, 
    h_edges::Diagonal             # D_e ∈ R^{e_id × e_id}     diagonal matrix of hyperedge degrees 
    weights::Diagonal             # D_w ∈ R^{e_id × e_id}     diagonal matrix of hyperedge weights
    v_neighbours::Dict{Int32,Set{Int32}} # the set of neighbours for each node
    Andp::U                              # A_ndp is equivalent to HRAdirect 
end

#
#Base.IndexStyle(::Type{<:myAbsHyperGraph}) = IndexLCartesian() # That's the default anyway so it does not say a lot...
# A rather resticted overloading, all indeces of a myHyperGraph refer solely to H, the Incidence matrix of the obj.
Base.size(A::myHyperGraph{T,U}) where {T,U} = size(A.H)
Base.size(A::myHyperGraph{T,U}, dim) where {T,U} = size(A.H, dim)
Base.length(A::myHyperGraph{T,U}) where {T,U} = length(A.H)
Base.getindex(A::myHyperGraph{T,U}, i::Int, j::Int) where {T,U} = get(A.H, (i, j), 0)
Base.getindex(A::myHyperGraph{T,U}, I::UnitRange, j::Int) where {T,U} = A.H[I, j]
Base.getindex(A::myHyperGraph{T,U}, i::Int, I) where {T,U} = [A.H[i, j] for j in I]
Base.getindex(A::myHyperGraph{T,U}, I, J) where {T,U} = A.H[I, J]

## ___________________________________________________
myHyperGraph(e, v) = myHyperGraph(e, v,
    zeros(Int32, v_id, e_id),
    Diagonal(zeros(Int32, 1)),
    Diagonal(zeros(Int32, 1)),
    Diagonal(ones(Float32, 1)),
    Dict{Int32,Set{Int32}}(),
    zeros(Float32, 1, 1)) # won't be used in this cstor, so leave it small

## ___________________________________________________
myHyperGraph(D::Diagonal{Int32}, E::Diagonal{Int32}) =
    myHyperGraph(size(E, 1),
        size(D, 1),
        zeros(Int32, size(D, 1), size(E, 1)),
        D,
        E,
        Diagonal(ones(Float32, size(E, 1))),
        Dict{Int64,Set{Int64}}(),
        zeros(Float32, 1, 1))
## ___________________________________________________
function myHyperGraph(D::Vector{Int32}, E::Vector{Int32})
    myHyperGraph(Diagonal(D), Diagonal(E))
end
## ___________________________________________________
#function myHyperGraph(h::T) where {T<:Union{Hypergraph,Matrix{Union{Nothing,Real}}}}
function myHyperGraph(h::T) where {T<:Union{Hypergraph,AbstractMatrix}}

    H = Incidence(h)
    D = nodes_degree_mat(h)
    E = hyperedges_degree_mat(h)
    eid = Int32(size(E, 1))
    nid = Int32(size(D, 1))
    W = hyper_weights(h)

    d = Dict{Int32,Set{Int32}}()
    for i in 1:nid
        d[i] = h_Neighbours(h, i)
    end

    Ei = E - I

    # The Inverse of Ei
    DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in eachindex(view(Ei, :, 1))] |> Diagonal)
    @info """In myHyperGraph(h::T) 
        Performing H * W * DeInv * H' - D, where
        size(E) =     $(size(E)),          typeof(E) =     $(typeof(E))
        size(Ei) =     $(size(Ei)),          typeof(Ei) =     $(typeof(Ei))
        size(H) =     $(size(H)),          typeof(H) =     $(typeof(H))
        size(W) =     $(size(W)),          typeof(W) =     $(typeof(W)),
        size(DeInv) = $(size(DeInv)),      typeof(DeInv) = $(typeof(DeInv)),
        size(D) =     $(size(D)),          typeof(D) =     $(typeof(D))
    """


    global size_of_change_2_sparse

    Andp = sparse_or_mutstat_mat(D)
    display(Andp)
    temporary = W * DeInv
    @show isdiag(temporary)
    Dtemp = Matrix{Float32}(undef, size(H, 1), size(temporary, 2)) |> sparse_or_mutstat_mat
    Dtemp = H * temporary
    Andp = Dtemp * H' - Andp

    # for some reason this get stuck
    #mul!(Andp, mul!(Dtemp, H, temporary, 1.0, 0), transpose(H), 1.0, -1.0)

    #mul!(Dtemp, H, temporary)
    @info "Dtemp $(size(Dtemp))"
    #mul!(Dtemp, H, temporary, 1, 0)
    #display(Dtemp)
    #@show typeof(Andp), typeof(H), typeof(H'), typeof(temporary), typeof(Dtemp)
    #Dtemp = Dtemp * H' - Dtemp
    #mul!(Andp, mul!(Dtemp, H, temporary, 1, 0), transpose(H), 1, -1)
    #Andp = H * W * DeInv * H' - D
    #display(Andp)
    #display(Andp1)
    #@assert sum(Andp) ≈ sum(Andp1)
    #@info "diff $(sum(Andp - Andp1))"
    println("Hypergraph edges=$(eid), nodes=$(nid)")
    @show typeof.((eid, nid, H, D, E, W, d, Andp))
    return myHyperGraph(eid, nid, H, D, E, W, d, Andp)

end

##############################################

node_density(hh::myHyperGraph) = sum(hh.H) / (hh.e_id * hh.v_id)
node_density(h::Hypergraph) = sum(Incidence(h)) / (length(h.he2v) * length(h.v2he))

##############################################

function nodes_degree_mat(h::Hypergraph)
    """Diagonal matrix of node degrees. Practically the same as node_dist"""

    # This...
    #D = Base.mapreduce(x -> (x != (0) && !isnothing(x)), +, h, dims=2)
    #Diagonal([D...])
    # and this...
    D = Diagonal(length.(h.v2he))
    # ...are the same

end
## ------------------------------------------------------
function nodes_degree_mat(h::SparseMatrixCSC)
    """Diagonal matrix of node degrees. Practically the same as node_dist"""

    # This...
    D = Base.mapreduce(x -> (x != (0) && !isnothing(x)), +, h, dims=2)
    Diagonal([D...])
    # and this...
    # D = Diagonal(length.(h.v2he))
    # ...are the same --- but a SparseMatrixCSC
    # does not have a v2he element!

 end
##############################################
function hyperedges_degree_mat(h)
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

###############################################
function hyper_weights(h)
    """
    Diagonal matrix of weights. For teh time being, just set them all equal to 1
    """
    Diagonal(ones(Float32, size(h, 2)))
end

###############################################
function sparse_or_static_mat(h::AbstractMatrix{T}; ssize=size_of_change_2_sparse)::AbstractMatrix where {T<:Number}
    #     if length(h) < 100
    #         return SMatrix{size(h)...}(Float64.(h))
    #     elseif 100 <= length(h) < ssize
    #         return Float64.(h)
    #     else
    #         return sparse(Float32.(h))
    #     end
    return sparse(Float32.(h))
end

function sparse_or_mutstat_mat(h::AbstractMatrix{T}; ssize=size_of_change_2_sparse)::AbstractMatrix where {T<:Number}
    # if length(h) < 100
    #     return MMatrix{size(h)...}(Float64.(h))
    # elseif 100 <= length(h) < ssize
    #     return Float64.(h)
    # else
    #     return sparse(Float32.(h))
    # end
    return sparse(Float32.(h))
end

################################################
function Incidence(h)

    hh = replace(h, nothing => 0.0, true => 1.0)
    # incidence may me a static matrix 
    sparse_or_static_mat(hh)
end
#const H(h) = Incidence(h)
###############################################


A(h::myHyperGraph) = h.H * h.weights * h.H' - h.nodes
# adjacency, SimpleHypergraphs already had one.
# Anyway, it's a cross check.



function A_ndp(h::myHyperGraph)
    Ei = h.h_edges - I
    DeInv = ([Ei[i, i] != 0.0 ? 1 / Ei[i, i] : 0.0 for i in eachindex(view(Ei, :, 1))] |> Diagonal)
    return h.H * h.weights * DeInv * h.H' - h.nodes
end

function A_ndp(h::Hypergraph)
    h |> myHyperGraph |> A_ndp
end

####################################################
function choose_new_vertex(hg, new_he)::Int64
    #calculate scores and put them in a vector 
    # This is defined to be of length eq to n of nodes in the hgraph, 
    # which is a bit too much, but this way we have an easy way to know 
    # which node had which score without the need for a dict, just by the node's index.
    # And vectors are quite fast.
    scores = calc_all_NHAS(hg, new_he)
    if sum(scores) < eps(Float64)
        return 0
    end

    # Choose a node (which is not in new_he) w/ prob proportional to its score 
    # -> the scores are weights of the sampling. so   
    # we can take only non zero scores (since 0 does not contribute anything)


    indeces = findall(!=(0.0), scores)
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

#############################################################

function create_new_hyperedge(hg::myHyperGraph; n::Int64=1)
"""
    create new random hyperedges for a ginve hypergraph
"""

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
        else
            # ... again start this hyperedge from the start 
            continue
        end

    end
    #@show length(new_hedge_size), length(hyperedges)
    return hyperedges
end
## -------------------------------------------------------------
function create_new_hyperedge(hg::myHyperGraph, kfold::T, j::T; n::Int64=1) where {T<:AbsatractVector{Int}}


    #create a vector to hold the output
    #hyperedges = Vector{HyperEdge}(undef, n)
    hyperedges = Vector{HyperEdge}()

    # we sample E^T, ie for hyperedges not in J
    new_hedges_sizes(i) = sample(diag(hg.h_edges)[kfold], i)

    kfoldnodes = nodes_degree_mat(hg[:, kfold]) |> diag

    function first_nodes(i, nodes)
        
        nodes_distribution = (nodes  |> fweights)
        return sample(1:hg.v_id, nodes_distribution, i)

    end

    while length(hyperedges) < n
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
HRA_direct(h::myHyperGraph, x::Int, y::Int) = h.Andp[x, y]
###--------------------------------------------------------------------------------------
function HRA_indirect(h::myHyperGraph, x::Int, y::Int) # n_commmon_edges::Int=1, edge_size::Int=1)
    L"""
        Inirect part of Hyper Resource Allocation.   
        HRA_{indirect} (x, y) =  \sum_{z ∈N (x) ∩ N(y)} 
        HRA_{direct} (x,z) × \frac{1}{d(z)}  × HRA_{direct} (z, y) 
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
        temp = h.Andp[x, z] / h.nodes[z, z]
        temp *= h.Andp[z, y]

        hra_ind += temp

    end

    return hra_ind
end
##-------------------------------------------------------------

HRA(h::myHyperGraph, x::Int, y::Int; α=0.5) = α * HRA_indirect(h, x, y) + (1 - α)h.Andp[x, y]

####################################################
function HRA_Matrix(h::myHyperGraph; α=0.5)
    """
        Calculates all pairwise HRA values for h
    """
    m = Matrix{Float64}(undef, h.v_id, h.v_id) |> UpperTriangular
    for i in 1:h.v_id
        for j in i:h.v_id

            m[i, j] = HRA(h, i, j, α=α)
            @debug "HRA_Matrix: i= $i  j=$j   m[i, j]=$(m[i, j])"
        end
    end
    return m
end

######################################################


function NHAS(h::myHyperGraph, vertex::Int, e_id::Int)
    """Node-Hyperedge Attachment Score.
    """

    res = 0.0
    for i in setdiff(findall(==(1), h.H[:, e_id]), [vertex])

        res += HRA(h, vertex, i)
        #@show i, HRA(HG, vertex, i), res
    end
    res /= h.h_edges[e_id]
end

# much slower
# @time begin
#     res2 = mapreduce(i -> HRA(HG, vertex, i), +, setdiff(findall(==(1), HG.H[:, j]), [vertex]))
#     res2 /= HG.h_edges[j]
# end

####################################################

function NHAS(h::myHyperGraph, vertex::Int, h_edge::HyperEdge)
    """Node-Hyperedge Attachment Score.
    """

    res = 0.0

    for i in setdiff(keys(h_edge.nodes), [vertex])

        res += HRA(h, vertex, i)
        #@show i, HRA(HG, vertex, i), res
    end
    res /= length(h_edge.nodes)
end


#

####################################################

function calc_all_NHAS(h::myHyperGraph, new_he::HyperEdge{T}) where {T<:Number}
    """Node-Hyperedge Attachment Score calculated for a Hypergraph 
    and an hyperedge which is not part of it.
    """

    # we can only calculate NHAS for nodes which are NOT part of new_he
    sdif = setdiff(1:h.v_id, keys(new_he.nodes))

    # Calculate NHAS scores
    sc = Folds.map(nod -> NHAS(h, nod, new_he), sdif)
end
#################################################################33

using Pipe: @pipe
find_empty_nodes(mat) = @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> any(==(0), _)
find_all_empty_nodes(mat) = @pipe replace(mat, nothing => 0) |> reduce(+, _, dims=2) |> findall(==(0), _)

#############################################################################

function find_connected_he(hyperg::myHyperGraph, cv_partition)
    """
        For a given partition cv_partition = (kfold, onefold) of myHyperGraph hyperg, check hyperg.H[kfold] (ie E^T)
        for rows (ie nodes) that sum to 0. These are nodes that do not exist in E^T, 
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

        # non_zero_hyperedges = @pipe replace(hyperg.H[x[1], onefold], nothing => 0) |> findall(!=(0), _)
        non_zero_hyperedges = findall(x -> x != (0) && x !=(nothing), hyperg.H[x[1], onefold])
        #println("Node $(x[1]) is zero in E^T. Checking E^M at $(non_zero_hyperedges)")
        push!(discarded_he, non_zero_hyperedges...)
        #display(hyperg[:, onefold])

    end

    # this HE set is discarded
    #discarded_he |> unique! |> sort!
    kept_he = setdiff(1:length(onefold), discarded_he)

    @info "Hyperedges #$(onefold[discarded_he]) discarded ------   we keep $(onefold[kept_he])"
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
function calc_av_F1score_matrix(Eᴾ::Matrix{Union{Nothing,T}}, Eᴹ::Matrix{Union{Nothing,U}}) where {T<:Real,U<:Real}
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

    for j in 1:p_dim
        for i in 1:m_dim

            fscore_matrix[i, j] = m(Eᴹ0[:, i], Eᴾ[:, j])
            #fscore_matrixReverse[i, j] = m(Eᴾ[:, i], Eᴹ0[:, j])
            fscore_matrix[i, j] == 0.0 ? nothing : @info "fscore_matrix[$i, $j] = $(fscore_matrix[i, j])"
            (sum(Eᴹ0[:, i]) != 0.0 && sum(Eᴾ[:, j]) != 0.0) ? nothing : @info "[$i, $j], sum(Eᴹ0[:, i])= $(sum(Eᴹ0[:, i]))  sum(Eᴾ[:, j])=$(sum(Eᴾ[:, j]))"
        end
    end
    #display(fscore_matrix)), 
    avg_F1 = 0.5 * (argmaxg_prime_mean(fscore_matrix) + argmaxg_mean(fscore_matrix))
    @debug "average F1  $(avg_F1)"
    return avg_F1 #, fscore_matrix #, fscore_matrixReverse
end

#############################################################################

function foldem(hyperg::myHyperGraph, fold_k)
    """
    Perform the k-fold cross validation. 
    After partitioning the h-edges of the h-graph into k subsets, we loop over them,
    identifying them as E^T (training) and E^M (missing) sets. 
    There may exist cases where E^T does not contain some node, ie contains empty hyperedges, 
    The relevant hyperedges (including the nodes) are not taken under consideration in E^M
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
        @info kept_hedges
        @debug hyperg.H[:, k]
        
        # If we create a new hypergraph we throw away a lot of resources, better do this in place
        #hhg = replace(hyperg.H[:, k], 0 => nothing) |> Hypergraph |> myHyperGraph

        # Now create new h-edges for the kfold E^T hgraph, that later we will 
        # compare to onefold E^M edges. 

        #new_Hedges = create_new_hyperedge(hhg, n=length(kept_hedges))
        new_Hedges = create_new_hyperedge(hhg, n=length(j))

        new_Hedges_mat = create_mat(new_Hedges)
        fs = calc_av_F1score_matrix(new_Hedges_mat, hyperg.H[:, kept_hedges])
        push!(av_f1_scores, fs)
        println("fs = $(fs)")
        println("###"^20)
        #print(stdout, ">>>> ")
        #read(stdin, 1)

    end
    return av_f1_scores #, fscore_matrix, fscore_matrixReverse
end



