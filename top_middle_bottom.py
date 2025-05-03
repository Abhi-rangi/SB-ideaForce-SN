import pandas as pd

# ① Load tables
sug = pd.read_csv("sbf_suggestion.csv", dtype={"suggestionId": int})
com = pd.read_csv("sbf_comment.csv",    dtype={"commentId": int, "suggestionId": int})

# ② Sort descending by votes
sorted_sug = sug.sort_values("votes", ascending=False, na_position="last").reset_index(drop=True)

# ③ Slice Top-500, Bottom-500, and true Middle-500
N = len(sorted_sug)
top500    = sorted_sug.iloc[:500].copy()
bottom500 = sorted_sug.sort_values("votes", ascending=True, na_position="last").head(500).copy()
start     = max(0, (N - 500) // 2)
middle500 = sorted_sug.iloc[start : start + 500].copy()

# ④ Tag each slice
top500   ["segment"] = "top"
middle500["segment"] = "middle"
bottom500["segment"] = "bottom"

# ⑤ Combine all suggestions
all_suggestions = pd.concat([top500, middle500, bottom500], ignore_index=True)

# ⑥ Drop large text fields
all_suggestions = all_suggestions.drop(columns=["body", "link"], errors="ignore")

# ⑦ Filter & tag comments by joining on suggestionId
#    This also brings in the 'segment' label for each comment
all_comments = (
    com.merge(
        all_suggestions[["suggestionId", "segment"]],
        on="suggestionId",
        how="inner"
    )
    .drop(columns=["body", "link"], errors="ignore")
)

# ⑧ (Optional) Export combined CSVs
all_suggestions.to_csv("combined_500_suggestions.csv", index=False)
all_comments.to_csv("combined_500_comments.csv",   index=False)

print("✓  Exported combined suggestions (500 top/mid/bottom) and their comments.")
