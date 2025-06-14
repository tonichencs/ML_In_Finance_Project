# %%
from data_utilities import *
# %%
# Data clean and preprocessing section
earning_calls_process(data_load('Predictors/earnings_calls-001.parquet'))
# %%
dataset = pq.ParquetDataset('earning_calls_full_part_000.parquet') 
table = dataset.read()
df = table.to_pandas()
example = df[:10]

# %%
process_and_save_embeddings(
    parquet_pattern='earning_calls_full_part_*.parquet',
    output_dir='text_embeddings_parts',
    text_column='full_text',
    device='cuda'
)