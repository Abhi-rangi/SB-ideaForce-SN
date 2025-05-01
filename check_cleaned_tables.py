import pandas as pd
from database import engine

print('==== sbf_comment (First 10 Rows) ====')
df_comment = pd.read_sql('SELECT * FROM sbf_comment', engine)
print(df_comment.head(10))
print('\nInfo:')
df_comment.info()

print('\n==== sbf_suggestion (First 10 Rows) ====')
df_suggestion = pd.read_sql('SELECT * FROM sbf_suggestion', engine)
print(df_suggestion.head(10))
print('\nInfo:')
df_suggestion.info() 