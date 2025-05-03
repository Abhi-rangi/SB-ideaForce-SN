import pandas as pd

# ---------------------------------------------------------------------------
# ①  Load tables (switch to pd.read_sql(...) if you’re pulling straight from MySQL)
# ---------------------------------------------------------------------------
sug = pd.read_csv("sbf_suggestion.csv", dtype={"suggestionId": int})
com = pd.read_csv("sbf_comment.csv",    dtype={"commentId": int, "suggestionId": int})

# ---------------------------------------------------------------------------
# ②  Identify the Top‑100 suggestions by vote count
# ---------------------------------------------------------------------------
top100 = (
    sug.sort_values("votes", ascending=False, na_position="last")
       .head(100)
       .copy()                                # keep a standalone slice
)

# ---------------------------------------------------------------------------
# ③  Trim unneeded text columns
#     ‑ For suggestions: drop `body` and `link` (huge free‑text fields)
#     ‑ For comments   : keep only the comment IDs that belong to those 100 ideas,
#                        then drop `body` and `link`
# ---------------------------------------------------------------------------
top100_suggestions = (
    top100.drop(columns=["body", "link"], errors="ignore")     # ignore in case CSV already stripped
)

top100_comments = (
    com.loc[com["suggestionId"].isin(top100_suggestions["suggestionId"])]
       .drop(columns=["body", "link"], errors="ignore")
)

# ---------------------------------------------------------------------------
# ④  Optional: write back to disk for downstream analysis
# ---------------------------------------------------------------------------
top100_suggestions.to_csv("top100_suggestions.csv", index=False)
top100_comments.to_csv("top100_comments.csv",   index=False)

print("✓  Top‑100 suggestions and their associated comments exported (without body/link).")
