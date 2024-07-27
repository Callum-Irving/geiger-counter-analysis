import numpy as np
import polars as pl
import scipy.stats as ss


# Want to test bananas vs no source
# method from: https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/paired-sample-t-test/

df_nosource = pl.read_csv("./data/no_source_20240621.csv")
df_bananas = pl.read_csv("./data/two_bananas_5mm.csv")

cpms_nosource = df_nosource["CPM"].to_numpy()
cpms_bananas = df_bananas["CPM"].to_numpy()

min_len = min(len(cpms_nosource), len(cpms_bananas))

cpms_nosource = cpms_nosource[:min_len]
cpms_bananas = cpms_bananas[:min_len]

diffs = cpms_bananas - cpms_nosource
diff_mean = np.mean(diffs)
diff_std = np.std(diffs, ddof=1)
t = diff_mean / (diff_std / np.sqrt(min_len))
print("t value:", t)

p = ss.t.sf(t, min_len - 1)
print("p value:", p)
