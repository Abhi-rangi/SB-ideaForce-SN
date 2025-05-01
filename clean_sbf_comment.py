import pandas as pd
from database import engine

def shape_data(df):
    print(df.shape)

# --- sbf_comment table ---
print('==== sbf_comment TABLE ====')
df_comment = pd.read_sql('SELECT * FROM sbf_comment', engine)

print('Sample Data (Before Cleaning):')
print(df_comment[['commentId', 'body']].head(10))

# Strip leading/trailing whitespace from 'body' column
df_comment['body'] = df_comment['body'].astype(str).str.strip()

print('\nSample Data (After Stripping Whitespace from "body"):')
print(df_comment[['commentId', 'body']].head(10))

print('\nFull Cleaned Data (first 10 rows):')
shape_data(df_comment)

print('\nData Info:')
df_comment.info()
print('\nNull Value Count:')
print(df_comment.isnull().sum())

def suggest_preprocessing(df, table_name):
    suggestions = []
    if df.isnull().any().any():
        suggestions.append('There are missing values. Consider filling or dropping nulls.')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].str.isspace().any() or df[col].str.startswith(' ').any() or df[col].str.endswith(' ').any():
            suggestions.append(f'Column "{col}" has leading/trailing whitespace. Consider stripping whitespace.')
    if 'timestamp' in df.columns:
        try:
            pd.to_datetime(df['timestamp'])
        except Exception:
            suggestions.append('Column "timestamp" may have invalid date formats. Consider parsing or cleaning.')
    if not suggestions:
        print(f'{table_name} is clean!')
    else:
        print(f'Issues found in {table_name}:')
        for s in suggestions:
            print('-', s)

# Check sbf_comment
suggest_preprocessing(df_comment, 'sbf_comment')

# --- sbf_suggestion table ---
print('\n\n==== sbf_suggestion TABLE ====')
df_suggestion = pd.read_sql('SELECT * FROM sbf_suggestion', engine)

print('Sample Data (Before Cleaning):')
print(df_suggestion[['suggestionId', 'body', 'title']].head(10))

# Clean text columns: body, title, author, link, category
for col in ['body', 'title', 'author', 'link', 'category']:
    if col in df_suggestion.columns:
        df_suggestion[col] = df_suggestion[col].astype(str).str.strip()

print('\nSample Data (After Stripping Whitespace from text columns):')
print(df_suggestion[['suggestionId', 'body', 'title']].head(10))

print('\nFull Cleaned Data (first 10 rows):')
shape_data(df_suggestion)

print('\nData Info:')
df_suggestion.info()
print('\nNull Value Count:')
print(df_suggestion.isnull().sum())

# Check sbf_suggestion
suggest_preprocessing(df_suggestion, 'sbf_suggestion')

# --- Overwrite cleaned data back to DB ---
try:
    df_comment.to_sql('sbf_comment', engine, if_exists='replace', index=False)
    print('\nSuccessfully overwrote sbf_comment table with cleaned data.')
except Exception as e:
    print(f'Failed to overwrite sbf_comment table: {e}')
    exit(1)

try:
    df_suggestion.to_sql('sbf_suggestion', engine, if_exists='replace', index=False)
    print('Successfully overwrote sbf_suggestion table with cleaned data.')
except Exception as e:
    print(f'Failed to overwrite sbf_suggestion table: {e}')
    exit(1)

print('\nAll preprocessing and DB overwrites completed successfully.') 