import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from community import best_partition  # python-louvain package
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import datetime
from lifelines import CoxPHFitter
from textblob import TextBlob  # For sentiment analysis
import warnings
warnings.filterwarnings('ignore')

# Database connection (replace with your actual connection details)
from sqlalchemy import create_engine
from database import engine

# Load data
df_suggestion = pd.read_sql('SELECT * FROM sbf_suggestion', engine)
df_comment = pd.read_sql('SELECT * FROM sbf_comment', engine)

print(f"Loaded {len(df_suggestion)} suggestions and {len(df_comment)} comments")

# Convert timestamps to datetime objects (handling the warning from earlier)
for df in [df_suggestion, df_comment]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Sort by timestamp to ensure chronological processing
df_suggestion = df_suggestion.sort_values('timestamp')
df_comment = df_comment.sort_values('timestamp')

# Let's assume we have a votes table or extract votes from suggestion table
# For this example, we'll use the 'votes' column in df_suggestion
# If you have a separate votes table, you'd load it here

# ------------------------------------
# 1. Network Construction
# ------------------------------------

def construct_bipartite_graph():
    """
    Construct a user-idea bipartite graph.
    Nodes are users and suggestions.
    Edges represent votes or comments, weighted by count.
    """
    B = nx.Graph()
    
    # Add suggestion nodes
    for _, row in df_suggestion.iterrows():
        B.add_node(f"s_{row['suggestionId']}", 
                  type='suggestion', 
                  title=row['title'],
                  category=row['category'],
                  votes=row['votes'],
                  timestamp=row['timestamp'])
    
    # Add user nodes from suggestions (authors)
    for _, row in df_suggestion.iterrows():
        B.add_node(f"u_{row['author']}", type='user')
        # Connect author to their suggestion
        B.add_edge(f"u_{row['author']}", f"s_{row['suggestionId']}", 
                  weight=1, type='authored')
    
    # Add user nodes from comments and connect to suggestions
    comment_counts = defaultdict(int)
    for _, row in df_comment.iterrows():
        user = f"u_{row['author']}"
        suggestion = f"s_{row['suggestionId']}"
        
        if not B.has_node(user):
            B.add_node(user, type='user')
        
        # Count comments from this user to this suggestion
        key = (user, suggestion)
        comment_counts[key] += 1
    
    # Add edges for comments with weights
    for (user, suggestion), count in comment_counts.items():
        if B.has_node(user) and B.has_node(suggestion):
            B.add_edge(user, suggestion, weight=count, type='commented')
    
    # Note: In a real implementation, you'd also add edges for votes
    # This would require a votes table or extracting that information
    
    print(f"Bipartite graph created with {len(B.nodes())} nodes and {len(B.edges())} edges")
    return B

def construct_user_user_graph(bipartite_graph):
    """
    Construct a user-user interaction graph.
    Directed edges from commenter/voter to suggestion's author.
    Edge weight is the number of interactions.
    """
    G = nx.DiGraph()
    
    # Get all user nodes
    users = [n for n, attr in bipartite_graph.nodes(data=True) if attr['type'] == 'user']
    for user in users:
        G.add_node(user)
    
    # Create dictionary mapping suggestions to their authors
    suggestion_to_author = {}
    for node in bipartite_graph.nodes(data=True):
        node_id, attrs = node
        if attrs['type'] == 'suggestion':
            # Find the author of this suggestion
            for neighbor in bipartite_graph.neighbors(node_id):
                neighbor_attrs = bipartite_graph.nodes[neighbor]
                if neighbor_attrs['type'] == 'user':
                    edge_data = bipartite_graph.get_edge_data(node_id, neighbor)
                    if edge_data.get('type') == 'authored':
                        suggestion_to_author[node_id] = neighbor
                        break
    
    # Create directed edges from commenters to authors
    interaction_counts = defaultdict(int)
    for user in users:
        # Get suggestions this user has interacted with
        for neighbor in bipartite_graph.neighbors(user):
            neighbor_attrs = bipartite_graph.nodes[neighbor]
            if neighbor_attrs['type'] == 'suggestion':
                edge_data = bipartite_graph.get_edge_data(user, neighbor)
                if edge_data.get('type') == 'commented':  # or 'voted'
                    # Get the author of the suggestion
                    author = suggestion_to_author.get(neighbor)
                    if author and author != user:  # Don't add self-loops
                        interaction_counts[(user, author)] += edge_data.get('weight', 1)
    
    # Add the weighted edges
    for (user, author), weight in interaction_counts.items():
        G.add_edge(user, author, weight=weight)
    
    print(f"User-User graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def construct_idea_idea_graph():
    """
    Construct an idea-idea similarity graph based on:
    1. Shared commenters
    2. Same category
    3. Content similarity (keywords)
    """
    G_ideas = nx.Graph()
    
    # Add suggestion nodes
    for _, row in df_suggestion.iterrows():
        G_ideas.add_node(f"s_{row['suggestionId']}", 
                        title=row['title'],
                        category=row['category'],
                        body=row['body'])
    
    # 1. Connect ideas that share commenters
    # Create a dictionary of {suggestion_id: set(commenters)}
    suggestion_commenters = defaultdict(set)
    for _, row in df_comment.iterrows():
        suggestion_commenters[f"s_{row['suggestionId']}"].add(row['author'])
    
    # Add edges for suggestions that share commenters
    for s1 in G_ideas.nodes():
        commenters1 = suggestion_commenters.get(s1, set())
        if not commenters1:
            continue
            
        for s2 in G_ideas.nodes():
            if s1 >= s2:  # Skip self-loops and duplicate edges
                continue
                
            commenters2 = suggestion_commenters.get(s2, set())
            if not commenters2:
                continue
                
            # Calculate Jaccard similarity of commenter sets
            shared_commenters = len(commenters1.intersection(commenters2))
            if shared_commenters > 0:
                jaccard = shared_commenters / len(commenters1.union(commenters2))
                G_ideas.add_edge(s1, s2, weight_commenters=jaccard)
    
    # 2. Connect ideas in the same category
    for s1 in G_ideas.nodes():
        cat1 = G_ideas.nodes[s1]['category']
        for s2 in G_ideas.nodes():
            if s1 >= s2:  # Skip self-loops and duplicate edges
                continue
                
            cat2 = G_ideas.nodes[s2]['category']
            if cat1 == cat2:
                if G_ideas.has_edge(s1, s2):
                    G_ideas[s1][s2]['weight_category'] = 1.0
                else:
                    G_ideas.add_edge(s1, s2, weight_category=1.0)
    
    # 3. Connect ideas with similar content
    # Extract title and body text
    suggestion_texts = []
    suggestion_ids = []
    for node_id, attrs in G_ideas.nodes(data=True):
        text = f"{attrs['title']} {attrs['body']}"
        suggestion_texts.append(text)
        suggestion_ids.append(node_id)
    
    # Calculate text similarity
    if suggestion_texts:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(suggestion_texts)
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Add edges for similar content (threshold = 0.2)
        for i in range(len(suggestion_ids)):
            for j in range(i+1, len(suggestion_ids)):
                if cosine_sim[i, j] > 0.2:  # Threshold for similarity
                    s1, s2 = suggestion_ids[i], suggestion_ids[j]
                    if G_ideas.has_edge(s1, s2):
                        G_ideas[s1][s2]['weight_content'] = cosine_sim[i, j]
                    else:
                        G_ideas.add_edge(s1, s2, weight_content=cosine_sim[i, j])
    
    # Combine weights into a single weight
    for u, v, attrs in G_ideas.edges(data=True):
        # Average the available weights
        weights = [val for key, val in attrs.items() if key.startswith('weight_')]
        G_ideas[u][v]['weight'] = sum(weights) / len(weights) if weights else 0
    
    print(f"Idea-Idea graph created with {len(G_ideas.nodes())} nodes and {len(G_ideas.edges())} edges")
    return G_ideas

# ------------------------------------
# 2. Community Detection
# ------------------------------------

def detect_communities(G):
    """
    Apply the Louvain algorithm to detect communities in the graph.
    Returns a dictionary mapping nodes to community IDs.
    """
    # Apply Louvain community detection
    partition = best_partition(G)
    
    # Count the number of communities
    communities = set(partition.values())
    print(f"Detected {len(communities)} communities")
    
    # Add community as node attribute
    nx.set_node_attributes(G, partition, 'community')
    
    return partition

def characterize_communities(G, partition):
    """
    Characterize communities based on roles, activity, and topics.
    """
    communities = set(partition.values())
    
    # Initialize data structures
    community_stats = {c: {
        'size': 0,
        'avg_degree': 0,
        'avg_betweenness': 0,
        'avg_pagerank': 0,
        'categories': defaultdict(int)
    } for c in communities}
    
    # Calculate centrality measures once
    degree_cent = dict(G.degree(weight='weight'))
    betweenness_cent = nx.betweenness_centrality(G, weight='weight', k=min(100, len(G)))
    pagerank = nx.pagerank(G, weight='weight')
    
    # Gather statistics for each community
    for node, comm_id in partition.items():
        # Update size
        community_stats[comm_id]['size'] += 1
        
        # Add centrality measures
        community_stats[comm_id]['avg_degree'] += degree_cent.get(node, 0)
        community_stats[comm_id]['avg_betweenness'] += betweenness_cent.get(node, 0)
        community_stats[comm_id]['avg_pagerank'] += pagerank.get(node, 0)
        
        # Extract node id from format "u_username" or "s_suggestionid"
        raw_id = node[2:] if node.startswith(('u_', 's_')) else node
        
        # Get categories of suggestions authored by this user
        if node.startswith('u_'):
            authored_suggestions = df_suggestion[df_suggestion['author'] == raw_id]
            for _, row in authored_suggestions.iterrows():
                community_stats[comm_id]['categories'][row['category']] += 1
    
    # Calculate averages
    for comm_id in communities:
        size = community_stats[comm_id]['size']
        if size > 0:
            community_stats[comm_id]['avg_degree'] /= size
            community_stats[comm_id]['avg_betweenness'] /= size
            community_stats[comm_id]['avg_pagerank'] /= size
        
        # Get top 3 categories for this community
        categories = community_stats[comm_id]['categories']
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        community_stats[comm_id]['top_categories'] = top_categories
    
    return community_stats

# ------------------------------------
# 3. Inter-Role Influence Analysis
# ------------------------------------

def analyze_role_influence(G):
    """
    Analyze influence between different roles in the network.
    """
    # Compute centrality measures
    print("Computing centrality measures...")
    degree_cent = dict(G.degree(weight='weight'))
    betweenness_cent = nx.betweenness_centrality(G, weight='weight', k=min(100, len(G)))
    pagerank = nx.pagerank(G, weight='weight')
    
    # Assign centrality as node attributes
    nx.set_node_attributes(G, degree_cent, 'degree_centrality')
    nx.set_node_attributes(G, betweenness_cent, 'betweenness_centrality')
    nx.set_node_attributes(G, pagerank, 'pagerank')
    
    # Identify roles based on activity level
    # Let's define roles based on percentiles of activity
    node_activities = {}
    for node in G.nodes():
        if node.startswith('u_'):
            # Extract username
            username = node[2:]
            
            # Count suggestions authored
            authored_count = len(df_suggestion[df_suggestion['author'] == username])
            
            # Count comments made
            comment_count = len(df_comment[df_comment['author'] == username])
            
            # Total activity
            activity = authored_count + comment_count
            node_activities[node] = activity
    
    # Define role thresholds
    if node_activities:
        activities = list(node_activities.values())
        expert_threshold = np.percentile(activities, 90)
        active_threshold = np.percentile(activities, 70)
        regular_threshold = np.percentile(activities, 30)
        
        # Assign roles
        roles = {}
        for node, activity in node_activities.items():
            if activity >= expert_threshold:
                roles[node] = 'expert'
            elif activity >= active_threshold:
                roles[node] = 'active'
            elif activity >= regular_threshold:
                roles[node] = 'regular'
            else:
                roles[node] = 'casual'
        
        # Add role as node attribute
        nx.set_node_attributes(G, roles, 'role')
        
        # Analyze role-to-role interactions
        role_interactions = defaultdict(int)
        for u, v, attrs in G.edges(data=True):
            source_role = roles.get(u, 'unknown')
            target_role = roles.get(v, 'unknown')
            weight = attrs.get('weight', 1)
            role_interactions[(source_role, target_role)] += weight
        
        return {
            'degree_centrality': degree_cent,
            'betweenness_centrality': betweenness_cent,
            'pagerank': pagerank,
            'roles': roles,
            'role_interactions': dict(role_interactions)
        }
    else:
        print("No user activity data available.")
        return None

# ------------------------------------
# 4. Idea Success & Lifetime Modeling
# ------------------------------------

def prepare_idea_features():
    """
    Engineer features for predicting idea success and lifetime.
    """
    # Join suggestions with comments
    merged_data = df_suggestion.copy()
    
    # Define success as having >= 10 votes
    merged_data['success'] = merged_data['votes'] >= 10
    
    # Calculate lifetime (days from first to last comment)
    idea_lifetimes = {}
    for suggestion_id in merged_data['suggestionId']:
        comments = df_comment[df_comment['suggestionId'] == suggestion_id]
        if len(comments) > 1:
            first_comment_time = comments['timestamp'].min()
            last_comment_time = comments['timestamp'].max()
            if pd.notna(first_comment_time) and pd.notna(last_comment_time):
                lifetime_days = (last_comment_time - first_comment_time).days
                idea_lifetimes[suggestion_id] = max(1, lifetime_days)  # Minimum 1 day
    
    merged_data['lifetime_days'] = merged_data['suggestionId'].map(idea_lifetimes)
    
    # Calculate early engagement metrics (first 24h)
    early_comments = {}
    early_vote_ratio = {}
    
    for suggestion_id, suggestion_time in zip(merged_data['suggestionId'], merged_data['timestamp']):
        if pd.isna(suggestion_time):
            continue
            
        # Early comments
        early_period_end = suggestion_time + pd.Timedelta(days=1)
        early_comment_count = len(df_comment[(df_comment['suggestionId'] == suggestion_id) & 
                                            (df_comment['timestamp'] <= early_period_end)])
        early_comments[suggestion_id] = early_comment_count
        
        # Early vote ratio (assuming votes accumulate over time)
        # Here we're using a proxy since we don't have the actual vote timestamps
        total_votes = merged_data.loc[merged_data['suggestionId'] == suggestion_id, 'votes'].values[0]
        if total_votes > 0:
            # Assume votes follow similar pattern as comments
            all_comments = len(df_comment[df_comment['suggestionId'] == suggestion_id])
            if all_comments > 0:
                ratio = early_comment_count / all_comments
                estimated_early_votes = total_votes * ratio
                early_vote_ratio[suggestion_id] = estimated_early_votes / total_votes
            else:
                early_vote_ratio[suggestion_id] = 0
        else:
            early_vote_ratio[suggestion_id] = 0
    
    merged_data['early_comments'] = merged_data['suggestionId'].map(early_comments)
    merged_data['early_vote_ratio'] = merged_data['suggestionId'].map(early_vote_ratio)
    
    # Add content features
    merged_data['title_length'] = merged_data['title'].str.len()
    merged_data['body_length'] = merged_data['body'].str.len()
    
    # Add sentiment analysis
    merged_data['sentiment'] = merged_data['body'].apply(
        lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0
    )
    
    # Fill missing values
    merged_data = merged_data.fillna({
        'early_comments': 0,
        'early_vote_ratio': 0,
        'lifetime_days': 1  # Default 1 day lifetime
    })
    
    # Select features for modeling
    feature_cols = [
        'early_comments', 'early_vote_ratio', 
        'title_length', 'body_length', 'sentiment',
        # Add any network features here once calculated
    ]
    
    # Filter to only include rows with complete data
    model_data = merged_data.dropna(subset=feature_cols + ['success'])
    
    print(f"Prepared features for {len(model_data)} suggestions")
    return model_data, feature_cols

def build_success_model(model_data, feature_cols):
    """
    Build a logistic regression model to predict suggestion success.
    """
    X = model_data[feature_cols]
    y = model_data['success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Calculate ROC AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Success Model - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}, ROC AUC: {roc_auc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'accuracy': test_score,
        'roc_auc': roc_auc,
        'feature_importance': feature_importance
    }

def build_lifetime_model(model_data, feature_cols):
    """
    Build a survival analysis model for suggestion lifetime.
    """
    # Filter to suggestions with lifetime data
    lifetime_data = model_data.dropna(subset=['lifetime_days'] + feature_cols)
    
    # Create event indicator (all 1's since we observed all lifetimes)
    lifetime_data['observed'] = 1
    
    # Prepare data for Cox Proportional Hazards model
    cph_data = lifetime_data[feature_cols + ['lifetime_days', 'observed']]
    
    # Fit model
    cph = CoxPHFitter()
    cph.fit(cph_data, duration_col='lifetime_days', event_col='observed')
    
    # Get summary
    summary = cph.summary
    
    print(f"Lifetime Model - Concordance Index: {cph.concordance_index_:.3f}")
    
    return {
        'model': cph,
        'concordance_index': cph.concordance_index_,
        'summary': summary
    }

# ------------------------------------
# Main execution
# ------------------------------------

def main():
    """
    Execute the full network analysis pipeline.
    """
    print("Starting Starbucks IdeaForce Network Analysis...")
    
    # 1. Construct Networks
    print("\n--- CONSTRUCTING NETWORKS ---")
    bipartite_graph = construct_bipartite_graph()
    user_user_graph = construct_user_user_graph(bipartite_graph)
    idea_idea_graph = construct_idea_idea_graph()
    
    # 2. Community Detection
    print("\n--- COMMUNITY DETECTION ---")
    partition = detect_communities(user_user_graph)
    community_stats = characterize_communities(user_user_graph, partition)
    
    # Print community summaries
    print("\nCommunity Characteristics:")
    for comm_id, stats in community_stats.items():
        print(f"Community {comm_id}: {stats['size']} members")
        print(f"  - Avg Degree: {stats['avg_degree']:.3f}")
        print(f"  - Avg Betweenness: {stats['avg_betweenness']:.3f}")
        print(f"  - Avg PageRank: {stats['avg_pagerank']:.5f}")
        print(f"  - Top Categories: {stats['top_categories']}")
    
    # 3. Inter-Role Influence Analysis
    print("\n--- ROLE INFLUENCE ANALYSIS ---")
    influence_results = analyze_role_influence(user_user_graph)
    
    if influence_results:
        # Print role distribution
        role_counts = defaultdict(int)
        for node, role in influence_results['roles'].items():
            role_counts[role] += 1
        
        print("\nRole Distribution:")
        for role, count in role_counts.items():
            print(f"{role}: {count} users")
        
        # Print centrality by role
        role_centrality = defaultdict(list)
        for node, role in influence_results['roles'].items():
            role_centrality[role].append({
                'degree': influence_results['degree_centrality'].get(node, 0),
                'betweenness': influence_results['betweenness_centrality'].get(node, 0),
                'pagerank': influence_results['pagerank'].get(node, 0)
            })
        
        print("\nAverage Centrality by Role:")
        for role, values in role_centrality.items():
            avg_degree = np.mean([v['degree'] for v in values])
            avg_betweenness = np.mean([v['betweenness'] for v in values])
            avg_pagerank = np.mean([v['pagerank'] for v in values])
            print(f"{role}:")
            print(f"  - Avg Degree: {avg_degree:.3f}")
            print(f"  - Avg Betweenness: {avg_betweenness:.5f}")
            print(f"  - Avg PageRank: {avg_pagerank:.5f}")
        
        # Print role interactions
        print("\nRole Interactions (weighted edges):")
        for (source, target), weight in influence_results['role_interactions'].items():
            print(f"{source} → {target}: {weight:.1f}")
    
    # 4. Idea Success & Lifetime Modeling
    print("\n--- IDEA SUCCESS & LIFETIME MODELING ---")
    model_data, feature_cols = prepare_idea_features()
    
    # Add network features to the feature set if needed
    # For example, author centrality in user-user network
    
    # Build and evaluate models
    success_results = build_success_model(model_data, feature_cols)
    
    print("\nSuccess Prediction - Feature Importance:")
    print(success_results['feature_importance'])
    
    # Build lifetime model
    lifetime_results = build_lifetime_model(model_data, feature_cols)
    
    print("\nLifetime Model - Summary:")
    print(lifetime_results['summary'])
    
    # Save networks for future use
    nx.write_gexf(bipartite_graph, "bipartite_graph.gexf")
    nx.write_gexf(user_user_graph, "user_user_graph.gexf")
    nx.write_gexf(idea_idea_graph, "idea_idea_graph.gexf")
    
    print("\nAnalysis complete. Networks saved as GEXF files.")

if __name__ == "__main__":
    main()