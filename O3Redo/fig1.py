import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

G = nx.read_gexf("O3Redo/results/co_commenter.gexf")

# use spring layout (slow on 9k nodes; precompute & cache)
pos = nx.spring_layout(G, k=0.08, seed=0)

# extract communities and degrees
comm = nx.get_node_attributes(G, "community")
deg  = dict(G.degree(weight="weight"))

# pick a colour per community
import matplotlib.cm as cm
palette = cm.get_cmap("tab10")
colors  = [palette(comm[n] % 10) for n in G]

# draw
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos,
                       node_size=[deg[n]*0.8 for n in G],
                       node_color=colors,
                       alpha=0.85, linewidths=0)
nx.draw_networkx_edges(G, pos,
                       edge_color="#cccccc22",
                       width=0.2, alpha=0.4)
plt.axis("off")
plt.tight_layout()
plt.savefig("O3Redo/figs/co_comm_coloured.png", dpi=300)
