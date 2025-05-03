#!/usr/bin/env python3
"""
Compute and print key network‐size metrics for the Starbucks IdeaForce data:
  Nₛ = number of suggestions
  N_c = number of comments
  N_u = number of users
  N_i = number of ideas
  E_b = edges in the bipartite graph
  T   = number of temporal snapshot periods (by month)
  F   = number of features
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
from sqlalchemy import create_engine
from database import engine

# --- Data Loading ---
df_suggestion = pd.read_sql('SELECT * FROM sbf_suggestion', engine)
df_comment    = pd.read_sql('SELECT * FROM sbf_comment', engine)

# --- Preprocess timestamps ---
for df in (df_suggestion, df_comment):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# --- 1. Base counts ---
N_s = len(df_suggestion)                      # number of suggestions
N_c = len(df_comment)                         # number of comments

users_s = set(df_suggestion['author'])
users_c = set(df_comment['author'])
N_u     = len(users_s.union(users_c))         # number of unique users

N_i = N_s                                    # ideas ≃ suggestions

# --- 2. Construct bipartite graph & count edges ---
def construct_bipartite_graph(sugg_df, comm_df):
    B = nx.Graph()
    # suggestion nodes
    for _, row in sugg_df.iterrows():
        B.add_node(f"s_{row['suggestionId']}")
    # author edges
    for _, row in sugg_df.iterrows():
        user = f"u_{row['author']}"
        sid  = f"s_{row['suggestionId']}"
        B.add_node(user)
        B.add_edge(user, sid)
    # comment edges
    counts = defaultdict(int)
    for _, row in comm_df.iterrows():
        key = (f"u_{row['author']}", f"s_{row['suggestionId']}")
        counts[key] += 1
    for (user, sid), c in counts.items():
        B.add_edge(user, sid, weight=c)
    return B

B  = construct_bipartite_graph(df_suggestion, df_comment)
E_b = B.number_of_edges()                     # bipartite edges

# --- 3. Compute T = number of monthly snapshot periods ---
# Use the suggestion timestamps to count unique Year-Month periods
periods = df_suggestion['timestamp'].dt.to_period('M')
T       = periods.nunique()

# --- 4. Feature count (from your modeling pipeline) ---
feature_cols = [
    'early_comments',
    'title_length',
    'body_length',
    'sentiment',
    'author_deg_cent',
    'author_pr'
]
F = len(feature_cols)

# --- Print all metrics ---
print(f"Nₛ (suggestions):        {N_s}")
print(f"N_c (comments):           {N_c}")
print(f"N_u (users):              {N_u}")
print(f"N_i (ideas):              {N_i}")
print(f"E_b (bipartite edges):    {E_b}")
print(f"T (snapshot periods):     {T}")
print(f"F (feature count):        {F}")
