# make_role_heatmap.py
import networkx as nx
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns          # seaborn is fine for static figs

# ------------------------------------------------------------------
# 1. Load co‑commenter graph with role attribute
# ------------------------------------------------------------------
G = nx.read_gexf("O3Redo/results/co_commenter.gexf")   # adjust path

roles = nx.get_node_attributes(G, "role")
role_set = ["expert", "contributor", "client"]  # fixed order

# ------------------------------------------------------------------
# 2. Build 3×3 mixing matrix Mij
# ------------------------------------------------------------------
M = pd.DataFrame(0, index=role_set, columns=role_set, dtype=int)

for u, v, d in G.edges(data=True):
    ru, rv = roles[u], roles[v]
    weight = d.get("weight", 1)
    M.loc[ru, rv] += weight
    if ru != rv:                         # undirected → symmetric
        M.loc[rv, ru] += weight

# optional: save counts
M.to_csv("O3Redo/results/role_mixing_matrix.csv")

# ------------------------------------------------------------------
# 3. Plot log‑scaled heat map
# ------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(np.log10(M.replace(0, np.nan)),      # log10, hide zeros
            annot=True, fmt=".1f", cmap="viridis",
            cbar_kws={"label": "log$_{10}$ edge weight"})
plt.title("Role-mixing matrix $M_{ij}$ for $G_{co}$")
plt.xlabel("Role of node $v$")
plt.ylabel("Role of node $u$")
plt.tight_layout()
plt.savefig("O3Redo/figs/role_heatmap.pdf", dpi=300)
plt.savefig("O3Redo/figs/role_heatmap.png", dpi=300)
plt.close()
