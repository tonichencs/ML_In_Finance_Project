import pandas as pd
import os
# %%

with open("Predictors/CompFirmCharac.csv/CompFirmCharac.csv", 'r', encoding='utf-8') as f:
    full_header = f.readline().strip().split(',')

selected_columns = [
    'datadate', 'gvkey', 'cusip', 'sic', 'oibdpy', 'capxy', 'invtq', 'actq', 'ancq',
    'ltq', 'lctq', 'niq', 'cogsq', 'revtq', 'chechy', 'cshfdy', 'cshpry', 'xintq',
    'txty', 'epspxy', 'dltry', 'dpcy', 'saleq', 'atq'
]
existing_columns = [col for col in selected_columns if col in full_header]
print(f"Columns to be loaded: {existing_columns}")


chunksize = 100_000
filtered_chunks = []

for chunk in pd.read_csv("Predictors/CompFirmCharac.csv/CompFirmCharac.csv", usecols=existing_columns, chunksize=chunksize, dtype={'cusip': str}, low_memory=False):
    # Convert date and filter by year
    chunk['datadate'] = pd.to_datetime(chunk['datadate'], errors='coerce')
    
    # Drop rows where date conversion failed right away
    chunk.dropna(subset=['datadate'], inplace=True)
    
    chunk = chunk[chunk['datadate'].dt.year >= 2000].copy()
    
    filtered_chunks.append(chunk)


print("Combining all chunks...")
df_filtered = pd.concat(filtered_chunks, ignore_index=True)
print(f"Shape after combining chunks and year filter: {df_filtered.shape}")

# %%

print("Sorting, dropping duplicates, and filling missing values...")

df_filtered = df_filtered.dropna(subset=['cusip', 'datadate', 'gvkey'])
df_filtered = df_filtered.drop_duplicates(subset=['cusip', 'datadate', 'gvkey'])

df_filtered = df_filtered.sort_values(['gvkey', 'datadate'])

fill_cols = ['capxy', 'chechy', 'cshfdy', 'cshpry', 'dltry', 'dpcy', 'epspxy', 'oibdpy', 'txty']
df_filtered[fill_cols] = df_filtered.groupby('gvkey')[fill_cols].ffill().bfill()

print("\nFinal data types:")
print(df_filtered.dtypes.to_string()) 

os.makedirs("data", exist_ok=True)
df_filtered.to_csv("data/filtered_compustat_char.csv", index=False)

print(f"\nFinal shape: {df_filtered.shape}")
print(df_filtered.head(3))

#%%
print(len(df_filtered['cusip'].unique()))
