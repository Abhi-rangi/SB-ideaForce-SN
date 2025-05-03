import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# --- connection setup (you already have this) ---
password = quote_plus("RoronovaZoro@3")
connection_string = f"mysql+pymysql://root:{password}@localhost:3306/set_local"
engine = create_engine(connection_string)

# --- read each table into a DataFrame ---
df_suggestions = pd.read_sql_table("sbf_suggestion", con=engine)
df_comments    = pd.read_sql_table("sbf_comment", con=engine)

# --- export to CSV ---
df_suggestions.to_csv("sbf_suggestion.csv", index=False)
df_comments   .to_csv("sbf_comment.csv",    index=False)

print("Exported sbf_suggestion.csv with shape", df_suggestions.shape)
print("Exported sbf_comment.csv with shape   ", df_comments.shape)
