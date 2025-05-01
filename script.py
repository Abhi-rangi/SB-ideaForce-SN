# database.py should be in the same directory or in your PYTHONPATH
from database import engine
import pandas as pd
import re

# Load tables from the database
sugg = pd.read_sql('SELECT * FROM sbf_suggestion', engine)
comm = pd.read_sql('SELECT * FROM sbf_comment', engine)

print("Raw sugg timestamps:", sugg['timestamp'].head(10).tolist())
print("Raw comm timestamps:", comm['timestamp'].head(10).tolist())

# Parse timestamps
for df, col in [(sugg, 'timestamp'), (comm, 'timestamp')]:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df.dropna(subset=[col], inplace=True)

# Normalize author names
def normalize_author(name):
    if pd.isnull(name):
        return ''
    return re.sub(r'\s+', '', name.strip().lower())

sugg['author_norm'] = sugg['author'].apply(normalize_author)
comm['author_norm'] = comm['author'].apply(normalize_author)

# Map author roles (heuristic)
def map_role(name):
    if 'expert' in name:
        return 'expert'
    elif 'starbucks' in name or 'admin' in name or 'moderator' in name:
        return 'firm_contributor'
    else:
        return 'regular_client'

sugg['role'] = sugg['author_norm'].apply(map_role)
comm['role'] = comm['author_norm'].apply(map_role)

# Merge roles into a master author-role mapping
author_roles = pd.concat([
    sugg[['author_norm', 'role']],
    comm[['author_norm', 'role']]
]).drop_duplicates().set_index('author_norm')['role'].to_dict()

print(sugg.head())
print(comm.head())
print(author_roles)
print(sugg.columns)
print(comm.columns)
print(len(sugg), "suggestions after timestamp parsing")
print(len(comm), "comments after timestamp parsing")