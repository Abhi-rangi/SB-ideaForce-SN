#!/usr/bin/env python3
"""
Optimized Starbucks IdeaForce Network Analysis Pipeline
- Vectorized builds via pandas
- Approximate idea–idea similarity via NearestNeighbors on active ideas
- Reduced dimensionality and neighbor count to prevent OOM
- Outputs .gexf for each major graph
- Logs start/end and node/edge counts for each phase
- Corrected user-user mapping via explicit node-type and schema-aware lookup
"""
import pandas as pd
import numpy as np
import networkx as nx
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from community import best_partition
from textblob import TextBlob
import warnings
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings('ignore')

# Database connection
def get_engine():
    from database import engine
    return engine

# ------------------------------------
# Setup & Data Loading
# ------------------------------------
print("[INIT] Starting Starbucks IdeaForce analysis pipeline...")
engine = get_engine()
df_sugg = pd.read_sql(
    'SELECT suggestionId, author, title, category, body, votes, timestamp FROM sbf_suggestion',
    engine
)
df_comm = pd.read_sql(
    'SELECT suggestionId, author, body, timestamp FROM sbf_comment',
    engine
)
print(f"[LOAD] Loaded {len(df_sugg)} suggestions and {len(df_comm)} comments")

# Preprocess timestamps
for df in (df_sugg, df_comm):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
print("[PREPROCESS] Converted timestamps to datetime.")

# Schema-aware node naming
# suggestion nodes: s_<suggestionId>
# user nodes: u_<author>
# Build mapping suggestion->author
sugg2auth = {f"s_{sid}": f"u_{auth}" for sid, auth in df_sugg[['suggestionId','author']].values}

# ------------------------------------
# 1. Bipartite Graph
# ------------------------------------
def construct_bipartite_graph():
    print("[PHASE 1] Constructing bipartite graph...")
    # Authorship edges
    auth = df_sugg[['author','suggestionId']].rename(columns={'author':'user','suggestionId':'idea'})
    auth['type'] = 'authored'; auth['weight'] = 1.0
    # Comment aggregation
    df_comm['sentiment'] = df_comm['body'].fillna('').apply(lambda t: TextBlob(t).sentiment.polarity)
    comm = df_comm.groupby(['author','suggestionId'], as_index=False).agg(
        count=('body','size'), avg_sent=('sentiment','mean')
    )
    comm.rename(columns={'author':'user','suggestionId':'idea'}, inplace=True)
    # Decay weighting
    now = pd.Timestamp.now()
    decay_vals = np.exp(-0.05 * (now - df_sugg.set_index('suggestionId')['timestamp']).dt.days)
    comm['decay'] = comm['idea'].map(decay_vals)
    comm['weight'] = comm['count'] * (1 + comm['avg_sent']) * comm['decay']
    comm['type'] = 'commented'
    # Combine edges
    edges = pd.concat([auth, comm[['user','idea','type','weight']]], ignore_index=True)
    # Build bipartite graph
    B = nx.Graph()
    # Add suggestion nodes
    for sid, row in df_sugg.set_index('suggestionId').iterrows():
        B.add_node(f"s_{sid}", node_type='suggestion', category=row['category'], votes=row['votes'], timestamp=row['timestamp'].isoformat())
    # Add user nodes
    for u in pd.unique(edges['user']):
        B.add_node(f"u_{u}", node_type='user')
    # Add edges
    for _, r in edges.iterrows():
        B.add_edge(f"u_{r.user}", f"s_{r.idea}", weight=float(r.weight), edge_type=r.type)
    # Export
    nx.write_gexf(B, 'bipartite_graph.gexf')
    print(f"[PHASE 1] Saved bipartite_graph.gexf ({B.number_of_nodes()} nodes, {B.number_of_edges()} edges)")
    return B

# ------------------------------------
# 2. User-User Graph
# ------------------------------------
def construct_user_user_graph(B):
    print("[PHASE 2] Constructing user-user graph...")
    G = nx.DiGraph()
    # Add user nodes
    user_nodes = [n for n, a in B.nodes(data=True) if a.get('type')=='user']
    G.add_nodes_from(user_nodes)
    print(f"[PHASE 2] Added {len(user_nodes)} user nodes")
    # Project comment edges
    for u, v, attrs in B.edges(data=True):
        if attrs.get('type')!='commented': continue
        # determine endpoints
        if B.nodes[u]['type']=='user': commenter = u; sug = v
        else: commenter = v; sug = u
        author = sugg2auth.get(sug)
        if author and author!=commenter:
            prev = G.get_edge_data(commenter, author, {'weight':0})['weight']
            G.add_edge(commenter, author, weight=prev + attrs['weight'])
    print(f"[PHASE 2] Added {G.number_of_edges()} directed edges")
    nx.write_gexf(G, 'user_user_graph.gexf')
    print(f"[PHASE 2] Saved user_user_graph.gexf ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G

# ------------------------------------
# 3. Idea-Idea Graph (Optimized)
# ------------------------------------
def construct_idea_graph(k=5):
    print("[PHASE 3] Constructing idea-idea graph (active ideas + ANN)...")
    # Filter to active ideas (some comments or votes)
    active_ids = set(df_comm['suggestionId']) | set(df_sugg[df_sugg['votes']>0]['suggestionId'])
    active = df_sugg[df_sugg['suggestionId'].isin(active_ids)].reset_index(drop=True)
    texts = (active['title'].fillna('') + ' ' + active['body'].fillna('')).tolist()
    # Lower TF-IDF dims
    tfidf = TfidfVectorizer(max_features=500, stop_words='english').fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1).fit(tfidf)
    dists, idxs = nn.kneighbors(tfidf)
    G = nx.Graph()
    # Add nodes
    for sid in active['suggestionId']:
        G.add_node(f"s_{sid}", category=active.loc[active['suggestionId']==sid,'category'].iat[0])
    # Add ANN edges
    for i, sid in enumerate(active['suggestionId']):
        src = f"s_{sid}"
        for dist, j in zip(dists[i][1:], idxs[i][1:]):
            if 1-dist>0.2:
                tgt = f"s_{active['suggestionId'].iat[j]}"
                w = 1-dist
                G.add_edge(src, tgt, weight_content=w)
    # Combine edges with simple average
    for u,v,attrs in G.edges(data=True):
        G[u][v]['weight'] = attrs['weight_content']
    nx.write_gexf(G, 'idea_idea_graph.gexf')
    print(f"[PHASE 3] Saved idea_idea_graph.gexf ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G

# ------------------------------------
# 4. Community Detection
# ------------------------------------
def detect_communities(G):
    print("[PHASE 4] Detecting communities on user graph...")
    # Convert to undirected as required by Louvain
    U_und = G.to_undirected()
    part = best_partition(U_und)
    # Apply partition labels back to the directed graph
    nx.set_node_attributes(G, part, 'community')
    print(f"[PHASE 4] Detected {len(set(part.values()))} communities")
    nx.write_gexf(G, 'user_user_graph_communities.gexf')
    print("[PHASE 4] Saved user_user_graph_communities.gexf")
    return part

# ------------------------------------
# Main
# ------------------------------------
def main():
    B = construct_bipartite_graph()
    U = construct_user_user_graph(B)
    I = construct_idea_graph()
    part = detect_communities(U)
    print("[DONE] Pipeline complete.")

if __name__=='__main__': main()
