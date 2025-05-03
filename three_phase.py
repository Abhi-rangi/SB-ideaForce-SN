import pandas as pd
import networkx as nx
from itertools import combinations

# --- 1. Extraction & Cleaning ---

# Load CSV exports
sug = pd.read_csv("sbf_suggestion.csv").rename(columns={"author":"author_sug"})
com = pd.read_csv("sbf_comment.csv").rename(columns={"author":"author_com"})

# Parse timestamps
sug["timestamp"] = pd.to_datetime(sug["timestamp"], infer_datetime_format=True)
com["timestamp"] = pd.to_datetime(com["timestamp"], infer_datetime_format=True)

# Drop rows with missing user or suggestionId
sug.dropna(subset=["suggestionId", "author_sug"], inplace=True)
com.dropna(subset=["suggestionId", "author_com"], inplace=True)

# --- 2. Node & Edge Construction ---

# A. Directed Comment Flow: edges from comment-author -> suggestion-author
merged = com[["suggestionId","author_com"]].merge(
    sug[["suggestionId","author_sug"]], on="suggestionId"
)
edge_flow = (
    merged
    .groupby(["author_com","author_sug"])
    .size()
    .reset_index(name="weight")
    .rename(columns={"author_com":"src","author_sug":"dst"})
)

# B. Undirected Co-commenter: users who commented the same suggestion
co_pairs = []
for sugg_id, group in com.groupby("suggestionId")["author_com"]:
    users = set(group)
    for u, v in combinations(users, 2):
        co_pairs.append((u, v))
df_co = pd.DataFrame(co_pairs, columns=["u","v"])
edge_co = (
    df_co
    .groupby(["u","v"])
    .size()
    .reset_index(name="weight")
)

# C. Suggestion Projection: suggestions linked by shared users (authors or commenters)
# Build user-suggestion mapping
sug_edges = pd.concat([
    sug[["suggestionId","author_sug"]].rename(columns={"author_sug":"user"}),
    com[["suggestionId","author_com"]].rename(columns={"author_com":"user"})
]).drop_duplicates()

proj_pairs = []
for user, group in sug_edges.groupby("user")["suggestionId"]:
    for s1, s2 in combinations(set(group), 2):
        proj_pairs.append((s1, s2))
df_proj = pd.DataFrame(proj_pairs, columns=["s1","s2"])
edge_proj = (
    df_proj
    .groupby(["s1","s2"])
    .size()
    .reset_index(name="weight")
)

# D. Bipartite User–Suggestion: user <-> suggestion edges (authorship or comment)
bip_edges = (
    sug_edges
    .groupby(["user","suggestionId"])
    .size()
    .reset_index(name="weight")
)

# --- 3. Graph Assembly & Analysis ---

# A. Directed Comment Flow Graph
G_flow = nx.DiGraph()
for _, row in edge_flow.iterrows():
    G_flow.add_edge(row.src, row.dst, weight=int(row.weight))

# B. Undirected Co-commenter Graph
G_co = nx.Graph()
for _, row in edge_co.iterrows():
    G_co.add_edge(row.u, row.v, weight=int(row.weight))

# C. Suggestion Projection Graph
G_proj = nx.Graph()
for _, row in edge_proj.iterrows():
    G_proj.add_edge(f"sug_{row.s1}", f"sug_{row.s2}", weight=int(row.weight))

# D. Bipartite Graph
G_bip = nx.Graph()
# add user nodes
for user in sug_edges["user"].unique():
    G_bip.add_node(f"user_{user}", bipartite="user")
# add suggestion nodes
for sug_id in sug["suggestionId"].unique():
    G_bip.add_node(f"sug_{sug_id}", bipartite="suggestion")
# add edges
for _, row in bip_edges.iterrows():
    G_bip.add_edge(f"user_{row.user}", f"sug_{row.suggestionId}", weight=int(row.weight))

# Export to GEXF for Gephi
nx.write_gexf(G_flow,   "comment_flow.gexf")
nx.write_gexf(G_co,     "co_commenters.gexf")
nx.write_gexf(G_proj,   "suggestion_projection.gexf")
nx.write_gexf(G_bip,    "user_suggestion_bipartite.gexf")

print("Graphs exported:")
print(" - comment_flow.gexf")
print(" - co_commenters.gexf")
print(" - suggestion_projection.gexf")
print(" - user_suggestion_bipartite.gexf")
