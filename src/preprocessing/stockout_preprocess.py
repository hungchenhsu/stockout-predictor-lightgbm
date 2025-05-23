# stockout_preprocess.py    
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 0. Parameters ----------
DATA_PATH = Path("features-daily_2024-11_dairy.csv")

ROLL_WINDOWS = [3, 7]      # you can tweak later
HOLIDAYS = pd.to_datetime(["2024-11-28", "2024-11-29", "2024-11-30"])

# ---------- 1. Load ----------
df = pd.read_csv(DATA_PATH)
df["daydate"] = pd.to_datetime(df["daydate"])
print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")

# ---------- 2. Basic EDA ----------
n_items = df["itemsku"].nunique()
print(f"Unique itemsku: {n_items:,}")

# identify items always zero
always_zero = (
    df.groupby("itemsku")["DailyBOH"]
      .apply(lambda s: (s == 0).all())
      .loc[lambda s: s]
      .index.tolist()
)
print(f"Items always zero in Nov: {len(always_zero):,}")

# ---------- 2b. Save permanent OOS list ----------
import json
PERM_OOS_PATH = Path("models/prod_v1/permanent_oos_list.json")

always_zero = (
    df.groupby("itemsku")["DailyBOH"]
      .apply(lambda s: (s == 0).all())
      .loc[lambda s: s]
      .index
      .astype(str)            # convert to string for consistency
      .tolist()
)

PERM_OOS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(PERM_OOS_PATH, "w") as f:
    json.dump({"generated_on": str(pd.Timestamp.today().date()),
               "sku_count": len(always_zero),
               "itemsku":   always_zero},
              f, indent=2)

print(f"Permanent OOS list saved → {PERM_OOS_PATH}  ({len(always_zero)} SKUs)")

# ---------- 3. Holiday flag ----------
df["is_holiday"] = df["daydate"].isin(HOLIDAYS).astype(int)

# ---------- 4. Create stock-out label ----------
def first_zero_date(s):
    zeros = s.index[s == 0]
    return zeros.min() if len(zeros) else pd.NaT

stockout_map = (
    df.groupby("itemsku")
      .apply(lambda g: first_zero_date(g.set_index("daydate")["DailyBOH"]))
)
df = df.merge(
    stockout_map.rename("oos_date"),
    on="itemsku",
    how="left"
)

# days_to_oos (positive integer; NaN if never stock-out within window)
df["days_to_oos"] = (df["oos_date"] - df["daydate"]).dt.days
df.loc[df["days_to_oos"] < 0, "days_to_oos"] = np.nan  # after OOS date → ignore

# ---------- 5. Rolling features ----------
df_sorted = df.sort_values(["itemsku", "daydate"])
for win in ROLL_WINDOWS:
    df_sorted[f"boh_mean_{win}"] = (
        df_sorted.groupby("itemsku")["DailyBOH"]
                 .transform(lambda x: x.rolling(win, min_periods=1).mean())
    )
    df_sorted[f"boh_std_{win}"] = (
        df_sorted.groupby("itemsku")["DailyBOH"]
                 .transform(lambda x: x.rolling(win, min_periods=1).std().fillna(0))
    )

# ---------- 6. Train / Test split ----------
train_end = pd.Timestamp("2024-11-25")
train_mask = df_sorted["daydate"] <= train_end
test_mask  = df_sorted["daydate"] > train_end

train_df = df_sorted[train_mask & df_sorted["days_to_oos"].notna() & ~df_sorted["itemsku"].isin(always_zero)]
test_df  = df_sorted[test_mask]

print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
print(f"Memory usage  (train): {train_df.memory_usage(deep=True).sum()/1e6:.2f} MB")

# Optional: save processed parquet for faster reload
train_df.to_parquet("train_proc.parquet", index=False)
test_df.to_parquet("test_proc.parquet", index=False)