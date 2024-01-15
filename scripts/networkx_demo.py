import networkx as nx
G = nx.Graph()

G.add_node(1)
G.add_nodes_from([2, 3])

G.add_nodes_from([
     (4, {"color": "red"}),
     (5, {"color": "green"}), 
])

H = nx.path_graph(10)
G.add_nodes_from(H)

G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)

G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(H.edges)

G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')

G.number_of_nodes(), G.number_of_edges()

list(G.nodes)
list(G.edges)
list(G.adj[1])  # or list(G.neighbors(1))

G.edges([2, 'm'])
G.degree([2, 3, 'm'])

G.remove_node(2)
list(G.nodes)

import matplotlib.pyplot as plt
G = nx.petersen_graph()
subax1 = plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')
subax2 = plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.show()


# Node Assortativity Coefficients and Correlation Measures https://networkx.org/nx-guides/content/algorithms/assortativity/correlation.html

