#!/usr/bin/env python3
"""
starbucks_network_pipeline.py
---------------------------------
End‑to‑end analysis to satisfy assignment Q1–Q4.

Usage:
    pip install pandas numpy networkx python-louvain scikit-learn
    python starbucks_network_pipeline.py
"""

import warnings, numpy as np, pandas as pd, networkx as nx
from pathlib import Path
from itertools import combinations
import community as community_louvain
from networkx.algorithms.community import quality as nxq
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0.  Paths & I/O
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR  = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1.  Load raw tables  (CSV or SQL)
# -----------------------------------------------------------------------------
# If you have direct DB access, uncomment & edit:
# import sqlalchemy as sa
# engine = sa.create_engine("mysql+pymysql://user:pass@localhost/db")
# sug = pd.read_sql("SELECT * FROM sbf_suggestion", engine)
# com = pd.read_sql("SELECT * FROM sbf_comment",     engine)

SUG_FILE = DATA_DIR / "combined_500_suggestions.csv"
COM_FILE = DATA_DIR / "combined_500_comments.csv"
sug = pd.read_csv(SUG_FILE)
com = pd.read_csv(COM_FILE)

# basic tidy
for df in (sug, com):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
sug.dropna(subset=["suggestionId", "author"], inplace=True)
com.dropna(subset=["commentId", "suggestionId", "author"], inplace=True)

# -----------------------------------------------------------------------------
# 2.  GRAPH CONSTRUCTION  (Q1)
# -----------------------------------------------------------------------------
print("Building global edge lists…")

# 2a  Comment‑flow  (directed, weighted)
edge_flow = (
    com.merge(sug[["suggestionId", "author"]], on="suggestionId",
              suffixes=("_com", "_sug"))
       .groupby(["author_com", "author_sug"])
       .size().reset_index(name="weight")
       .query("author_com != author_sug")
)

# 2b  Co‑commenter  (undirected, weighted)
pairs = (
    com[["suggestionId", "author"]].drop_duplicates()
       .merge(com[["suggestionId", "author"]], on="suggestionId")
       .query("author_x < author_y")
       .value_counts(["author_x", "author_y"])
       .reset_index(name="weight")
       .rename(columns={"author_x":"u", "author_y":"v"})
)

# 2c  User‑Idea bipartite (undirected, unweighted)
ui_edges = pd.concat([
        sug[["author", "suggestionId"]].rename(columns={"author":"user", "suggestionId":"idea"}),
        com[["author", "suggestionId"]].rename(columns={"author":"user", "suggestionId":"idea"})
    ], ignore_index=True).drop_duplicates()

# 2d  Idea‑projection (undirected, weighted)
sugg_pairs = (
    ui_edges.merge(ui_edges, on="user")
            .query("idea_x < idea_y")
            .value_counts(["idea_x","idea_y"])
            .reset_index(name="weight")
            .rename(columns={"idea_x":"s1","idea_y":"s2"})
)

# 2e  **Topic‑specific user–user graphs**  (one per category)
topic_edges = {}
for cat, ids in sug.groupby("category")["suggestionId"]:
    sub = com[com["suggestionId"].isin(ids)][["suggestionId","author"]].drop_duplicates()
    t_pairs = (sub.merge(sub, on="suggestionId")
                    .query("author_x < author_y")
                    .value_counts(["author_x","author_y"])
                    .reset_index(name="weight"))
    topic_edges[cat] = t_pairs

# -----------------------------------------------------------------------------
# 3.  BUILD NetworkX OBJECTS
# -----------------------------------------------------------------------------
G_flow = nx.DiGraph()
G_flow.add_weighted_edges_from(edge_flow[["author_com","author_sug","weight"]].values)

G_co = nx.Graph()
G_co.add_weighted_edges_from(pairs[["u","v","weight"]].values)

G_bip = nx.Graph()
G_bip.add_nodes_from(ui_edges["user"].unique(), bipartite="user")
G_bip.add_nodes_from(ui_edges["idea"].unique(), bipartite="idea")
G_bip.add_edges_from(ui_edges[["user","idea"]].values)

G_proj = nx.Graph()
G_proj.add_weighted_edges_from(sugg_pairs[["s1","s2","weight"]].values)

topic_graphs = {cat: nx.Graph() for cat in topic_edges}
for cat, df in topic_edges.items():
    topic_graphs[cat].add_weighted_edges_from(df[["author_x","author_y","weight"]].values)

# -----------------------------------------------------------------------------
# 4.  ROLE TAGGING  (heuristic)   (Q3 pre‑req)
# -----------------------------------------------------------------------------
activity = com["author"].value_counts()
vote_tot = sug.groupby("author")["votes"].sum()
act_cut  = activity.quantile(0.90)
vote_cut = vote_tot.quantile(0.95)

def role(u):
    lu = str(u).lower()
    if lu.startswith(("sbx", "starbucks_")):
        return "expert"
    if activity.get(u,0) >= act_cut or vote_tot.get(u,0) >= vote_cut:
        return "contributor"
    return "client"

roles = {u: role(u) for u in G_co.nodes()}
nx.set_node_attributes(G_co, roles, "role")

# -----------------------------------------------------------------------------
# 5.  COMMUNITY DETECTION  (Q2)
# -----------------------------------------------------------------------------
part_co = community_louvain.best_partition(G_co, weight="weight")
nx.set_node_attributes(G_co, part_co, "community")

# per‑topic communities (optionally saved)
topic_comm = {cat: community_louvain.best_partition(g, weight="weight")
              for cat, g in topic_graphs.items()}

Q_co = nxq.modularity(G_co,
                      [{n for n,c in part_co.items() if c==k} for k in set(part_co.values())],
                      weight="weight")

# intra vs inter traffic
intra_w = sum(d["weight"] for u,v,d in G_co.edges(data=True) if part_co[u]==part_co[v])
total_w = sum(d["weight"] for _,_,d in G_co.edges(data=True))
pct_intra = intra_w / total_w

# role assortativity
assort = nx.attribute_assortativity_coefficient(G_co,"role")

# -----------------------------------------------------------------------------
# 6.  IDEA‑LEVEL FEATURES & RIDGE LOGIT  (Q4)
# -----------------------------------------------------------------------------
n_comments = com.groupby("suggestionId").size()

idea_df = (sug.assign(n_comments = sug["suggestionId"].map(n_comments).fillna(0))
               .assign(author_bt = lambda d: d["author"].map(nx.betweenness_centrality(G_co, weight="weight")),
                       author_comm = lambda d: d["author"].map(part_co),
                       author_role = lambda d: d["author"].map(roles))
)

# success label = top‑decile **by votes**  → don't use raw votes in predictors!
idea_df["success"] = (idea_df["votes"] >= idea_df["votes"].quantile(0.90)).astype(int)

# predictors
idea_df["log_comments"] = np.log1p(idea_df["n_comments"])
idea_df["bt_z"] = StandardScaler().fit_transform(idea_df[["author_bt"]])

numeric = ["log_comments", "bt_z"]
categorical = ["category"]

X_train, X_test, y_train, y_test = train_test_split(
    idea_df[numeric + categorical], idea_df["success"],
    test_size=0.30, stratify=idea_df["success"], random_state=42
)

pre = ColumnTransformer(
    [("num", StandardScaler(), numeric),
     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
)

model = Pipeline([
    ("prep", pre),
    ("clf", LogisticRegression(penalty="l2",
                               solver="liblinear",
                               max_iter=500,
                               class_weight="balanced"))
])

cv = GridSearchCV(model, {"clf__C":[0.01,0.1,1,10]}, cv=5,
                  scoring="roc_auc", n_jobs=-1)
cv.fit(X_train, y_train)

bestC = cv.best_params_["clf__C"]
train_auc = roc_auc_score(y_train, cv.predict_proba(X_train)[:,1])
test_auc  = roc_auc_score(y_test,  cv.predict_proba(X_test)[:,1])

# -----------------------------------------------------------------------------
# 7.  EXPORTS
# -----------------------------------------------------------------------------
nx.write_gexf(G_flow, OUT_DIR / "comment_flow.gexf")
nx.write_gexf(G_co,   OUT_DIR / "co_commenter.gexf")
nx.write_gexf(G_bip,  OUT_DIR / "user_idea_bipartite.gexf")
nx.write_gexf(G_proj, OUT_DIR / "suggestion_projection.gexf")
for cat, g in topic_graphs.items():
    nx.write_gexf(g, OUT_DIR / f"topic_{cat[:15].replace(' ','_')}.gexf")

idea_df.to_csv(OUT_DIR / "idea_features.csv", index=False)

with open(OUT_DIR / "ridge_logit_summary.txt","w") as f:
    f.write(f"Best C      : {bestC}\n")
    f.write(f"Train AUC   : {train_auc:.3f}\n")
    f.write(f"Test  AUC   : {test_auc:.3f}\n")

# -----------------------------------------------------------------------------
# 8.  DASHBOARD
# -----------------------------------------------------------------------------
print("="*60)
print("PIPELINE COMPLETE  —  KEY NUMBERS")
print(f"Users (nodes, G_co)           : {G_co.number_of_nodes()}")
print(f"Comment‑flow edges            : {G_flow.number_of_edges()}")
print(f"Modularity (G_co)             : {Q_co:.3f}")
print(f"Intra‑community traffic       : {pct_intra:.2%}")
print(f"Role assortativity (G_co)     : {assort:.3f}")
print(f"Ridge Logit AUC  (train/test) : {train_auc:.2f} / {test_auc:.2f}")
print("Outputs in:", OUT_DIR.resolve())
print("="*60)
