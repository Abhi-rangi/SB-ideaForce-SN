import pandas as pd
import networkx as nx
from database import engine

# --- Load cleaned data ---
sugg = pd.read_sql('SELECT * FROM sbf_suggestion', engine)
comm = pd.read_sql('SELECT * FROM sbf_comment', engine)

# Normalize author names (same as in script.py)
def normalize_author(name):
    if pd.isnull(name):
        return ''
    import re
    return re.sub(r'\s+', '', name.strip().lower())

sugg['author_norm'] = sugg['author'].apply(normalize_author)
comm['author_norm'] = comm['author'].apply(normalize_author)

# Map author roles (same heuristic as before)
def map_role(name):
    if 'expert' in name:
        return 'expert'
    elif 'starbucks' in name or 'admin' in name or 'moderator' in name:
        return 'firm_contributor'
    else:
        return 'regular_client'

sugg['role'] = sugg['author_norm'].apply(map_role)
comm['role'] = comm['author_norm'].apply(map_role)

author_roles = pd.concat([
    sugg[['author_norm', 'role']],
    comm[['author_norm', 'role']]
]).drop_duplicates().set_index('author_norm')['role'].to_dict()

# --- 1. User–Idea Bipartite Graph ---
B = nx.Graph()

# Add user nodes (with role attribute)
for user, role in author_roles.items():
    B.add_node(user, bipartite='user', role=role)

# Add idea nodes
for sid in sugg['suggestionId']:
    B.add_node(f'suggestion_{sid}', bipartite='idea')

# Add edges for suggestions (author → idea)
for _, row in sugg.iterrows():
    B.add_edge(row['author_norm'], f'suggestion_{row["suggestionId"]}'), {'type': 'suggestion'}

# Add edges for comments (commenter → idea)
for _, row in comm.iterrows():
    B.add_edge(row['author_norm'], f'suggestion_{row['suggestionId']}'), {'type': 'comment'}

# Optionally, add vote edges if vote data is available (not shown here)

# --- 2. User–User Interaction Graph ---
G = nx.DiGraph()

# Add users as nodes (with role attribute)
for user, role in author_roles.items():
    G.add_node(user, role=role)

# Add directed edges: commenter/voter → suggestion's author
# From comments
for _, row in comm.iterrows():
    commenter = row['author_norm']
    # Find suggestion's author
    suggestion_row = sugg[sugg['suggestionId'] == row['suggestionId']]
    if not suggestion_row.empty:
        author = suggestion_row.iloc[0]['author_norm']
        if commenter != author:
            if G.has_edge(commenter, author):
                G[commenter][author]['weight'] += 1
            else:
                G.add_edge(commenter, author, weight=1, type='comment')

# From votes (if available)
# (Not implemented here; add similar logic if vote data is available)

# --- 3. Idea–Idea Similarity Graph (optional) ---
H = nx.Graph()

# Add idea nodes
for sid in sugg['suggestionId']:
    H.add_node(sid)

# Connect ideas sharing commenters
from collections import defaultdict
idea_commenters = defaultdict(set)
for _, row in comm.iterrows():
    idea_commenters[row['suggestionId']].add(row['author_norm'])

idea_ids = sugg['suggestionId'].tolist()
for i, id1 in enumerate(idea_ids):
    for id2 in idea_ids[i+1:]:
        shared = idea_commenters[id1] & idea_commenters[id2]
        if shared:
            H.add_edge(id1, id2, weight=len(shared), type='shared_commenter')

# Optionally, connect by category or keywords (not shown)

# --- Save or export graphs as needed ---
# nx.write_gpickle(B, 'user_idea_bipartite.gpickle')
# nx.write_gpickle(G, 'user_user_interaction.gpickle')
# nx.write_gpickle(H, 'idea_idea_similarity.gpickle')

# --- Example: print basic stats ---
print(f'User–Idea bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges')
print(f'User–User interaction graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
print(f'Idea–Idea similarity graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges') 