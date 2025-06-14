# %%
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import gc
import glob

# %%
# Data Loading
def data_load(filepath):
    pf = pq.ParquetFile(filepath)
    dfs = []
    
    for i in range(pf.num_row_groups):
        dfs.append(pf.read_row_groups([i]).to_pandas())
        
    final_df = pd.concat(dfs)
    return final_df


#%%
# Data Processing
def process_earning_calls_chunk(chunk):
    return (
        chunk.groupby(['Date', 'transcriptid', 'gvkey'], sort=False)['componenttext']
             .agg(' '.join)
             .reset_index(name='full_text')
             .set_index('Date')
    )

def earning_calls_process(df):
    df.rename(columns={'mostimportantdateutc': 'Date'}, inplace=True)
    df.drop(['companyname', 'transcriptcomponenttypename'], axis=1, inplace=True)
    df = df.sort_values(by=['Date', 'transcriptid','componentorder'], ascending=True)


    # Merge earning_calls into one single report for each Date and Company
    chunk_results = []
    # Parameters
    chunk_size = 100000

    # Iterate over chunks
    for start in range(0, len(df), chunk_size):
        print(start)
        end = start + chunk_size
        chunk = df.iloc[start:end]

        # Important: If a transcriptid is split across chunks, merge won't be correct.
        # So we'll buffer extra rows that share the same transcriptid at the boundary
        if end < len(df):
            # Get the transcriptid at the chunk end
            last_transcriptid = df.iloc[end - 1]['transcriptid']
            while end < len(df) and df.iloc[end]['transcriptid'] == last_transcriptid:
                end += 1
            chunk = df.iloc[start:end]

        # Process and collect the chunk
        processed = process_earning_calls_chunk(chunk)
        chunk_results.append(processed)
        
        del chunk
        del processed
        gc.collect()

    # Combine all processed chunks
    final_df = pd.concat(chunk_results).sort_index()
    
    del df
    
    # Write to new parquet files
    chunk_size = 100000  # or whatever fits your memory
    output_prefix = 'earning_calls_full_part'

    for i, start in enumerate(range(0, len(final_df), chunk_size)):
        end = min(start + chunk_size, len(final_df))
        df_chunk = final_df.iloc[start:end].copy()

        # Sanitize types for pyarrow compatibility
        df_chunk['gvkey'] = df_chunk['gvkey'].astype('string')
        for col in df_chunk.select_dtypes(include='object').columns:
            if col != 'gvkey':
                df_chunk[col] = df_chunk[col].astype('string')

        file_name = f'{output_prefix}_{i:03d}.parquet'

        # Convert and write parquet file
        table_chunk = pa.Table.from_pandas(df_chunk, preserve_index=True)
        pq.write_table(table_chunk, file_name, compression='snappy')
        print(f"Wrote rows {start} to {end - 1} into {file_name}")

        # Explicit cleanup
        del df_chunk
        del table_chunk
        gc.collect()
    
    return final_df

def load_earning_calls_full_text(parquet_pattern, text_column):
    files = sorted(glob.glob(parquet_pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files found matching pattern: {parquet_pattern}")
    
    df_list = []
    for file in files:
        df = pd.read_parquet(file, columns=[text_column])  # Load only needed column
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)
