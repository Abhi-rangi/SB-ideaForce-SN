#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# mystarbucks_full_pipeline.py
#
# Build complex‑network layers, detect communities, analyse role‑based ties,
# and model idea success / lifetime for the Top‑100 My Starbucks Idea dataset.
#
# Inputs
#   ├─ top100_suggestions.csv   (from sbf_suggestion)
#   └─ top100_comments.csv      (from sbf_comment)
#
# Outputs → ./outputs/
#   ├─ *.gexf  (four graph layers)
#   ├─ community_sizes_*.csv
#   ├─ inter_edges_co.csv
#   ├─ role_mixing_co.csv
#   ├─ centrality_by_role.csv
#   ├─ idea_features.csv
#   ├─ logit_summary.txt
#   └─ cox_summary.txt  (if timestamps for implementation exist)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 0.  Imports
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy  as np
import networkx as nx
from pathlib import Path
from itertools import combinations
import community              as community_louvain     # python‑louvain
from networkx.algorithms.community import quality      # modularity
import statsmodels.formula.api as smf

# Optional: survival analysis (lifelines)
try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# --------------------------------------------------------------------------- #
# 1.  Paths & I/O
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SUG_FILE = DATA_DIR / "top100_suggestions.csv"
COM_FILE = DATA_DIR / "top100_comments.csv"
ROLE_MAP = DATA_DIR / "user_roles.csv"          # optional mapping file

dtype_id = {"suggestionId": int, "commentId": int}

print("Loading CSVs...")
sug = pd.read_csv(SUG_FILE, dtype=dtype_id)
com = pd.read_csv(COM_FILE, dtype=dtype_id)

# --------------------------------------------------------------------------- #
# 2.  Minimal cleansing
# --------------------------------------------------------------------------- #
for df in (sug, com):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

sug.dropna(subset=["suggestionId", "author"], inplace=True)
com.dropna(subset=["commentId", "suggestionId", "author"], inplace=True)

# --------------------------------------------------------------------------- #
# 3.  Edge lists
# --------------------------------------------------------------------------- #
print("Building edge lists...")

# 3a. Comment‑flow (directed, weighted)
edge_flow = (
    com.merge(sug[["suggestionId", "author"]],
              on="suggestionId",
              suffixes=("_com", "_sug"))
       .groupby(["author_com", "author_sug"])
       .size()
       .reset_index(name="weight")
       .query("author_com != author_sug")
)

# 3b. Co‑commenter (undirected, weighted)
pairs = (
    com[["suggestionId", "author"]]
      .drop_duplicates()
      .merge(com[["suggestionId", "author"]], on="suggestionId")
      .query("author_x < author_y")
      .value_counts(["author_x", "author_y"])
      .reset_index(name="weight")
      .rename(columns={"author_x": "u", "author_y": "v"})
)

# 3c. User‑Idea bipartite (undirected, unweighted)
ui_edges = pd.concat(
    [
        sug[["author", "suggestionId"]].rename(
            columns={"author": "user", "suggestionId": "idea"}),
        com[["author", "suggestionId"]].rename(
            columns={"author": "user", "suggestionId": "idea"})
    ],
    ignore_index=True
).drop_duplicates()

# 3d. Suggestion projection (undirected, weighted)
sugg_pairs = (
    ui_edges.merge(ui_edges, on="user")
            .query("idea_x < idea_y")
            .value_counts(["idea_x", "idea_y"])
            .reset_index(name="weight")
            .rename(columns={"idea_x": "s1", "idea_y": "s2"})
)

# --------------------------------------------------------------------------- #
# 4.  Build graphs
# --------------------------------------------------------------------------- #
print("Constructing graphs...")

G_flow = nx.DiGraph()
G_flow.add_weighted_edges_from(edge_flow[["author_com", "author_sug", "weight"]].values)

G_co = nx.Graph()
G_co.add_weighted_edges_from(pairs[["u", "v", "weight"]].values)

G_bip = nx.Graph()
G_bip.add_nodes_from(ui_edges["user"].unique(), bipartite="user")
G_bip.add_nodes_from(ui_edges["idea"].unique(), bipartite="idea")
G_bip.add_edges_from(ui_edges[["user", "idea"]].values)

G_proj = nx.Graph()
G_proj.add_weighted_edges_from(sugg_pairs[["s1", "s2", "weight"]].values)

# --------------------------------------------------------------------------- #
# 4b.  Role tagging
# --------------------------------------------------------------------------- #
print("Tagging roles...")

def heuristic_role(username: str) -> str:
    """Fallback tagging if no explicit role mapping is supplied."""
    name = str(username).lower()
    if name.startswith("sbx"):                 # Starbucks employee marker
        return "expert"
    # toy heuristic: very active commenters = contributors
    return "contributor" if activity.get(username, 0) >= contributor_cut else "client"

# — activity counts used by heuristic
activity = com["author"].value_counts().to_dict()
contributor_cut = int(np.percentile(list(activity.values()), 90))  # top 10 %

role_dict = {}
if ROLE_MAP.exists():
    role_df = pd.read_csv(ROLE_MAP)          # expects columns: user, role
    role_dict = dict(zip(role_df["user"], role_df["role"]))

# fallback to heuristic for users not in mapping
for u in G_co.nodes():
    role_dict.setdefault(u, heuristic_role(u))

nx.set_node_attributes(G_co, role_dict, "role")

# --------------------------------------------------------------------------- #
# 5.  Community detection & quality
# --------------------------------------------------------------------------- #
print("Running Louvain…")
part_co   = community_louvain.best_partition(G_co,   weight="weight")
part_proj = community_louvain.best_partition(G_proj, weight="weight")

nx.set_node_attributes(G_co,   part_co,   "community")
nx.set_node_attributes(G_proj, part_proj, "community")

print("Calculating modularity…")
Q_co   = quality.modularity(G_co,   [ {n for n,c in part_co.items()   if c==k} for k in set(part_co.values()) ],   weight="weight")
Q_proj = quality.modularity(G_proj, [ {n for n,c in part_proj.items() if c==k} for k in set(part_proj.values()) ], weight="weight")

# --------------------------------------------------------------------------- #
# 6.  Analytics tables
# --------------------------------------------------------------------------- #
print("Generating analytics…")

# 6a. Community sizes
comm_sizes = (
    pd.Series(part_co)
      .value_counts()
      .rename_axis("community")
      .reset_index(name="size")
      .sort_values("community")
)
comm_sizes["modularity_G_co"] = Q_co
comm_sizes.to_csv(OUT_DIR / "community_sizes_co.csv", index=False)

# 6b. Inter‑community edge weights
edges_inter = [
    {"c1": min(part_co[u], part_co[v]),
     "c2": max(part_co[u], part_co[v]),
     "weight": d["weight"]}
    for u, v, d in G_co.edges(data=True) if part_co[u] != part_co[v]
]
pd.DataFrame(edges_inter)\
  .groupby(["c1", "c2"])["weight"].sum()\
  .reset_index()\
  .to_csv(OUT_DIR / "inter_edges_co.csv", index=False)

# 6c. Role mixing matrix & assortativity
roles      = nx.get_node_attributes(G_co, "role")
role_set   = sorted(set(roles.values()))
mix_mtx    = pd.DataFrame(0, index=role_set, columns=role_set, dtype=int)
for u, v in G_co.edges():
    mix_mtx.loc[roles[u], roles[v]] += 1
mix_mtx.to_csv(OUT_DIR / "role_mixing_co.csv")

assort = nx.attribute_assortativity_coefficient(G_co, "role")

# 6d. Centrality by role
print("Computing centralities…")
btw  = nx.betweenness_centrality(G_co, weight="weight")
deg  = dict(G_co.degree(weight="weight"))

cent_rows = []
for u in G_co.nodes():
    cent_rows.append({
        "user": u,
        "role": roles[u],
        "degree_w": deg[u],
        "betweenness": btw[u],
        "community": part_co[u]
    })
centrality_df = pd.DataFrame(cent_rows)
centrality_df.sort_values(["role", "betweenness"], ascending=[True, False])\
             .to_csv(OUT_DIR / "centrality_by_role.csv", index=False)

# --------------------------------------------------------------------------- #
# 7.  Idea‑level feature table & modelling
# --------------------------------------------------------------------------- #
print("Building idea feature set…")

# Comment counts
n_comments = com.groupby("suggestionId").size().rename("n_comments")

# Author centralities (lookup)
author_btw  = {u: btw.get(u, 0) for u in sug["author"]}
author_deg  = {u: deg.get(u, 0) for u in sug["author"]}
author_comm = {u: part_co.get(u, -1) for u in sug["author"]}
author_role = {u: roles.get(u, "unknown") for u in sug["author"]}

idea_df = sug.copy()
idea_df = idea_df.merge(n_comments, left_on="suggestionId", right_index=True, how="left")
idea_df["n_comments"].fillna(0, inplace=True)
idea_df["author_betweenness"] = idea_df["author"].map(author_btw)
idea_df["author_degree_w"]    = idea_df["author"].map(author_deg)
idea_df["author_community"]   = idea_df["author"].map(author_comm)
idea_df["author_role"]        = idea_df["author"].map(author_role)

# Binary success label: implemented flag if column present; else top‑decile by votes
if "implemented" in idea_df.columns:
    idea_df["success"] = idea_df["implemented"].astype(int)
else:
    top_decile = idea_df["votes"].quantile(0.90)
    idea_df["success"] = (idea_df["votes"] >= top_decile).astype(int)

idea_df.to_csv(OUT_DIR / "idea_features.csv", index=False)

# ----- Logistic regression --------------------------------------------------
print("Fitting logistic regression…")
formula = "success ~ votes + n_comments + author_betweenness + C(category)"
logit_model = smf.logit(formula, data=idea_df).fit(disp=False)
with open(OUT_DIR / "logit_summary.txt", "w") as f:
    f.write(logit_model.summary().as_text())

# ----- Survival / Cox PH model (optional) -----------------------------------
if HAS_LIFELINES and {"created_ts", "implemented_ts"}.issubset(idea_df.columns):
    print("Fitting Cox model…")
    idea_df["duration"] = (pd.to_datetime(idea_df["implemented_ts"])
                            - pd.to_datetime(idea_df["created_ts"])).dt.days
    idea_df.dropna(subset=["duration"], inplace=True)
    cox = CoxPHFitter()
    cox.fit(
        idea_df[["duration", "success", "votes", "n_comments", "author_betweenness"]],
        duration_col="duration",
        event_col="success"
    )
    with open(OUT_DIR / "cox_summary.txt", "w") as f:
        f.write(cox.summary.to_string())
else:
    print("Cox model skipped (lifelines not installed or timestamp columns missing).")

# --------------------------------------------------------------------------- #
# 8.  Export graphs
# --------------------------------------------------------------------------- #
print("Writing GEXF layers…")
nx.write_gexf(G_flow, OUT_DIR / "comment_flow.gexf")
nx.write_gexf(G_co,   OUT_DIR / "co_commenter.gexf")
nx.write_gexf(G_bip,  OUT_DIR / "user_idea_bipartite.gexf")
nx.write_gexf(G_proj, OUT_DIR / "suggestion_projection.gexf")

# --------------------------------------------------------------------------- #
# 9.  Final report
# --------------------------------------------------------------------------- #
print("=" * 60)
print("Pipeline complete.  Key stats")
print("- Number of users           :", G_co.number_of_nodes())
print("- Number of comments edges  :", G_flow.number_of_edges())
print(f"- Modularity (G_co)         : {Q_co:.3f}")
print(f"- Role assortativity (G_co) : {assort:.3f}")
print("- Logistic LL / Pseudo‑R²   :", logit_model.llf, "/", logit_model.prsquared)
print("Outputs saved to:", OUT_DIR.resolve())
print("=" * 60)
