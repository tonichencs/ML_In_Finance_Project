import pandas as pd
import os
# %%


# --- 1. Get Header and Define Columns (Your code is fine here) ---
with open("Predictors/CompFirmCharac.csv/CompFirmCharac.csv", 'r', encoding='utf-8') as f:
    full_header = f.readline().strip().split(',')

selected_columns = [
    'datadate', 'gvkey', 'cusip', 'sic', 'oibdpy', 'capxy', 'invtq', 'actq', 'ancq',
    'ltq', 'lctq', 'niq', 'cogsq', 'revtq', 'chechy', 'cshfdy', 'cshpry', 'xintq',
    'txty', 'epspxy', 'dltry', 'dpcy', 'saleq', 'atq'
]
existing_columns = [col for col in selected_columns if col in full_header]
print(f"Columns to be loaded: {existing_columns}")


# --- 2. Load data in chunks and do INITIAL filtering ---
chunksize = 100_000
filtered_chunks = []

# FIX: Added dtype={'cusip': str} to the read_csv call
for chunk in pd.read_csv("Predictors/CompFirmCharac.csv/CompFirmCharac.csv", usecols=existing_columns, chunksize=chunksize, dtype={'cusip': str}, low_memory=False):
    # Convert date and filter by year
    chunk['datadate'] = pd.to_datetime(chunk['datadate'], errors='coerce')
    
    # Drop rows where date conversion failed right away
    chunk.dropna(subset=['datadate'], inplace=True)
    
    chunk = chunk[chunk['datadate'].dt.year >= 2000].copy()
    
    filtered_chunks.append(chunk)


# --- 3. Combine chunks BEFORE doing complex operations ---
print("Combining all chunks...")
df_filtered = pd.concat(filtered_chunks, ignore_index=True)
print(f"Shape after combining chunks and year filter: {df_filtered.shape}")

# %%


# --- 4. Perform Sorting, Deduplication, and Filling on the FULL DataFrame ---
# This is the CORRECT place for these operations

print("Sorting, dropping duplicates, and filling missing values...")

# Optional: Drop duplicates and nulls in key columns
df_filtered = df_filtered.dropna(subset=['cusip', 'datadate', 'gvkey'])
df_filtered = df_filtered.drop_duplicates(subset=['cusip', 'datadate', 'gvkey'])

# Sort by firm and time
df_filtered = df_filtered.sort_values(['gvkey', 'datadate'])

# Forward-fill and backward-fill by firm
fill_cols = ['capxy', 'chechy', 'cshfdy', 'cshpry', 'dltry', 'dpcy', 'epspxy', 'oibdpy', 'txty']
df_filtered[fill_cols] = df_filtered.groupby('gvkey')[fill_cols].ffill().bfill()

# Optional: Now you can drop any remaining NaN rows if you wish
# df_filtered = df_filtered.dropna()


# --- 5. Save Output ---
print("\nFinal data types:")
print(df_filtered.dtypes.to_string()) # .to_string() shows all dtypes

os.makedirs("data", exist_ok=True)
df_filtered.to_csv("data/filtered_compustat_char.csv", index=False)

print(f"\nFinal shape: {df_filtered.shape}")
print(df_filtered.head(3))

#%%
print(len(df_filtered['cusip'].unique()))
