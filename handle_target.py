import pandas as pd
import numpy as np


# Original dataset
csv_dir = "Targets/monthly_crsp.csv/monthly_crsp.csv"
# # A relatively small dataset for testing
# test_dir = "./data/test.csv"

df = pd.read_csv(csv_dir, dtype={'CUSIP': str, 'HdrCUSIP': str})

# %%

df = df[['PERMNO', 'HdrCUSIP', 'CUSIP', 'Ticker', 'NAICS', 'MthCalDt', 'MthRet', 'sprtrn']]
df.columns = ['permno', 'hdrcusip', 'cusip', 'ticker', 'industry', 'date', 'ret', 'ret_market']

# %%

df_clean = df.dropna(subset=['ret', 'ret_market'], how='any').copy()

df_clean['date'] = pd.to_datetime(df_clean['date'])
df_clean = df_clean[df_clean['date'] >= '2000-01-01'].copy()
df_clean = df_clean.drop_duplicates(
    subset=['permno', 'date', 'cusip'], keep='first').copy()

# %%


def cum_forward(arr, n):
    values = np.asarray(arr)
    length = len(values)
    result = np.full(length, np.nan, dtype=float)
    for i in range(length - n + 1):
        result[i] = np.prod(values[i: i + n]) - 1
    return pd.Series(result, index=arr.index)
# %%

df = df_clean

df['ret_now_sign'] = df.apply(
    lambda row: (
        1 if (row['ret'] > 0)
        else (-1 if row['ret'] < 0
              else np.nan)
    ),
    axis=1
)

df['ret_lag1'] = df.groupby('permno')['ret'].shift(1)
df['ret_trend1_sign'] = df.apply(
    lambda row: (
        1 if (pd.notna(row['ret_lag1']) and row['ret'] > row['ret_lag1'])
        else (-1 if (pd.notna(row['ret_lag1']) and row['ret'] < row['ret_lag1'])
              else np.nan)
    ),
    axis=1
)

df.loc[:, 'ret_cum_forward3m'] = df.groupby('permno').apply(
    lambda grp: cum_forward(grp['ret'] + 1, 3),
    include_groups=False
).reset_index(level=0, drop=True)
df['ret_cum_forward3m_sign'] = df.apply(
    lambda row: (
        1 if (row['ret_cum_forward3m'] > 0)
        else (-1 if row['ret_cum_forward3m'] < 0
              else np.nan)
    ),
    axis=1
)

df.loc[:, 'ret_cum_forward12m'] = df.groupby('permno').apply(
    lambda grp: cum_forward(grp['ret'] + 1, 12),
    include_groups=False
).reset_index(level=0, drop=True)
df['ret_cum_forward12m_sign'] = df.apply(
    lambda row: (
        1 if (row['ret_cum_forward12m'] > 0)
        else (-1 if row['ret_cum_forward12m'] < 0
              else np.nan)
    ),
    axis=1
)

def cum_backward(arr, n):
    values = np.asarray(arr)
    length = len(values)
    result = np.full(length, np.nan, dtype=float)
    for i in range(n, length):
        result[i] = np.prod(values[i-n: i]) - 1
    return pd.Series(result, index=arr.index)


df.loc[:, 'ret_cum_backward3m'] = df.groupby('permno').apply(
    lambda grp: cum_backward(grp['ret'] + 1, 3),
    include_groups=False
).reset_index(level=0, drop=True)
df.loc[:, 'ret_cum_backward6m'] = df.groupby('permno').apply(
    lambda grp: cum_backward(grp['ret'] + 1, 6),
    include_groups=False
).reset_index(level=0, drop=True)
df.loc[:, 'ret_cum_backward12m'] = df.groupby('permno').apply(
    lambda grp: cum_backward(grp['ret'] + 1, 12),
    include_groups=False
).reset_index(level=0, drop=True)

# %%


df.shape
# %%


df.to_csv('data/target_data.csv', index=False)

# %%

df.head()