                                
                                                    
#################################################################33
function node_degree(h::Hypergraph, v_id::Int; edge_size::Int = 1)
    """Degree of a node: it is the number of edges it is contained within. 
    The optional edge_size parameter places a restriction on the size of the edges 
    you consider (default edge_size=1). The degree function looks 
    for all edges of size ≥  edge_size.
    """
    degr = 0.;
    eds = gethyperedges(h,v_id)
    for (e, w) in eds
            #println("vertex $v_id is in edge $e with weight $(w):")
            vs = getvertices(h,e)
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
    if degr > 0. 
        return degr
    end
    return nothing
end


#################################################################33
function edge_degree(h::Hypergraph, e_id::Int) #; edge_size::Int = 1)
    """Degree of a hyperedge e, its degree is defined as δ (e) = |e|
    """

    vs = getvertices(h,e_id)
    return length(vs) 
end
    

##################################################################3
function h_Neighbours(h::Hypergraph, v_id::Int; n_commmon_edges::Int = 1)
    """ a set containing the one-hop neighbors of node
    v_id (nodes of hyperedges, v is part of). The size keyword argument 
    returns the set of neighbors that share at least s edges with the given node, the default is 1.
    """
    eds = gethyperedges(h,v_id)
    neighb = Set()
    for (e, w) in eds
        vs = getvertices(h,e)
        #println(vs)
        if length(vs) >= n_commmon_edges
            for (v,ww) in vs
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
    res_edg = [];
    #res_vert = [];
    for dct in h.v2he
        #@show dct
        foreach(x->push!(res_edg, x.first), dct)
    
    end
    #@show length(res_vert), sort(unique(res_edg))
    return res_edg |> unique |> sort   
end

############################################################3
function h_nodes(h::Hypergraph )  #; number_format::Bool = true, ttor::Dict = ttor)
    # size(h) returns (n_vertices, n_edges)
    # h.v2he is a vector of the hyperedges ie , it does what the following loop woudl do: 
    # for ed_n in 1:size(h)[1] #gethyperedges(h,v_id)
    #     # gethyperedges(h,ed_n) returns the dict of the edges 
    #     # that contain vert er_n with the respecthve weights
    #     eds =  gethyperedges(h,ed_n)
    #     @show  eds  
    # end
    res_vert = [];
    for dct in h.he2v
        #@show dct
        #if number_format
       foreach(x->push!(res_vert, x.first), dct)
    #     else
    #         foreach(x->push!(res_vert, ttor[x.first]), dct)
    #     end
    # end
    #@show length(res_vert), sort(unique(res_edg))
    return res_vert |> unique  
end


###########################################
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
    
    ny = h_Neighbours(h, y, n_commmon_edges=n_commmon_edges)
    ra = 0.
    for z in nx ∩ ny
        ra += 1/node_degree(h, z, edge_size= edge_size) 
        #println("---- ",z, "  ",node_degree(h, z, edge_size= edge_size) )
    end
    return ra
end


function HRA_direct(h::Hypergraph, x::Int, y::Int ; n_commmon_edges::Int = 1, edge_size::Int = 1)
    L"""
    Direct part of Hyper Resource Allocation.   
    HRA_{direct} (x, y) = \sum_{e, s.t. x,y ∈e} 1/ [δ(e) - 1] 
    (is this equal to A_ndp?) 
    """
    if x == y
        return nothing
    end
    # first we need the hedges than include both x an y
    edsx = ( gethyperedges(h,x) |> keys )
    edsy = ( gethyperedges(h,y) |> keys )

    common_edges = intersect(edsx, edsy)
    if common_edges == Set()
        return nothing
    end

    hrad = 0.
    for ed in common_edges
        hr = edge_degree(h, ed) - 1 
        hrad += 1/hr        
        
    end
   return hrad
end

function HRA_indirect(h::Hypergraph, x::Int, y::Int ; n_commmon_edges::Int = 1, edge_size::Int = 1)
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

    hra_ind = 0.
    for z in zs
        
        temp =  node_degree(h, z)
        temp = HRA_direct(h,x,z) / temp
        temp *= HRA_direct(h,z,y) 

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
    direc = isnothing(temp) ? 0. : temp

    temp = HRA_indirect(h, x, y)
    indirec = isnothing(temp) ? 0. : temp
    return  direc + indirec
end

#####################################################
    
function NHAS(h::Hypergraph, x::Int, e_id::Int)

    nhas = 0.
    for (y, weight) in getvertices(h,e_id) # this is a dict, we only care for the key
        #@show y,weight, nhas
        temp = HRA(h,x, y)
        #@show temp
        if isnothing(temp)
            continue
        end
        nhas += temp 
    end

    nhas /= edge_degree(h, e_id)

end


#####################################################
 
mutable struct Spl <: Sampleable{Univariate, Discrete} 
    vect::Vector{Float64}
end

#####################################################
 


function hyperedge_dist(h::Hypergraph)
    """Returns a Distributions sampler, made after the distribution of
        hyperedge size of h
    """
    v = length.(h.v2he)
    return Spl(v)
end

function Base.rand(spl::Spl)
    l = length(spl.vect)
    spl.vect[rand(1:l)]
end

function hyperedge_distrib(h::Hypergraph)
    """Creates a histogram of the 
    #histogram(y, normalize=:pdf, label="rand")
    """
    length.(h.v2he)
end

#####################################################
 