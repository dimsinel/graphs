using DrWatson
@quickactivate "graphs"

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
show_embedding(
    y;
    A = adjacency_matrix(g),        # show edges on embedding
    mrk_size = 1,                   # control node sizes
    lwd_in = 0.01,
    lwd_out = 0.001, # control edge widths
    edge_alpha = 0.03,
)

using GraphMakie
using GraphMakie.NetworkLayout
g = smallgraph(:dodecahedral)
graphplot(g; layout = Stress(; dim = 3))


using MLUtils

# kfolds(n::Integer, k = 5) -> Tuple

# X is a matrix of floats
# Y is a vector of strings
X, Y = load_iris()

# The iris dataset is ordered according to their labels,
# which means that we should shuffle the dataset before
# partitioning it into training- and test-set.
Xs, Ys = shuffleobs((X, Y))

# We leave out 15 % of the data for testing
cv_data, test_data = splitobs((Xs, Ys); at = 0.85)

# Next we partition the data using a 10-fold scheme.
for (train_data, val_data) in kfolds(cv_data; k = 10)

    # We apply a lazy transform for data augmentation
    train_data = mapobs(xy -> (xy[1] .+ 0.1 .* randn.(), xy[2]), train_data)

    for epoch = 1:10
        # Iterate over the data using mini-batches of 5 observations each
        for (x, y) in eachobs(train_data, batchsize = 5)
            # ... train supervised model on minibatches here
            @show x, y
        end
    end
end
