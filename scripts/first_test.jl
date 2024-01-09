using DrWatson
@quickactivate "graphs"
a
using Graphs
# Here you may include files from the source directory
include(srcdir("dummy_src_file.jl"))



g = path_graph(6)

# Number of vertices
nv(g)

# Number of edges
ne(g)

# Add an edge to make the path a loop
add_edge!(g, 1, 6);

add_vertex!(g)

using GLMakie, SGtSNEpi, SNAPDatasets

GLMakie.activate!()

g = loadsnap(:as_caida)
y = sgtsnepi(g);
show_embedding(y;
  A = adjacency_matrix(g),        # show edges on embedding
  mrk_size = 1,                   # control node sizes
  lwd_in = 0.01, lwd_out = 0.001, # control edge widths
  edge_alpha = 0.03 ) 

using GraphMakie
using GraphMakie.NetworkLayout
g = smallgraph(:dodecahedral)
graphplot(g; layout=Stress(; dim=3))