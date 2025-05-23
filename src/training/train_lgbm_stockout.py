# train_lgbm_stockout.py    
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score
from lightgbm import LGBMRegressor
import joblib
import yaml

# ---------- 0. Parameters ----------
TRAIN_PATH = Path("train_proc.parquet")
TEST_PATH  = Path("test_proc.parquet")
MODEL_OUT  = Path("lgbm_stockout_model.txt")

# ---------- 1. Load ----------
train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)

# permanent OOS list (saved during preprocessing)
permanent_oos_list = (
    train_df["itemsku"].unique().tolist() +
    test_df.loc[test_df["DailyBOH"] == 0, "itemsku"].unique().tolist()
)
permanent_oos_list = list(set(permanent_oos_list))

# ---------- 2. Feature selection ----------
EXCLUDE = {
    "daydate", "oos_date", "days_to_oos"
}
num_feats = [c for c in train_df.columns
             if c not in EXCLUDE and
                train_df[c].dtype != "object" and
                not c.startswith("itemsku")]

CAT_FEATS = ["itemsku"]

FEATURES = num_feats + CAT_FEATS

# ensure categorical dtype
for c in CAT_FEATS:
    train_df[c] = train_df[c].astype("category")
    test_df[c]  = test_df[c].astype("category")

X = train_df[FEATURES]
y = train_df["days_to_oos"]

# ---------- 3. Model ----------
lgbm_params = dict(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    metric="mae",
)

model = LGBMRegressor(**lgbm_params)

# ---------- 4. Cross-validation ----------
gkf = GroupKFold(n_splits=5)
cv_mae = -cross_val_score(
    model, X, y,
    cv=gkf.split(X, y, groups=train_df["itemsku"]),
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
print(f"CV MAE (5-fold, grouped by itemsku): {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

# ---------- 5. Fit on full train ----------
model.fit(
    X, y,
    categorical_feature=CAT_FEATS,
)

# ---------- 6. Evaluate on hold-out (11/26–30) ----------
test_eval = test_df[test_df["days_to_oos"].notna()].copy()

if len(test_eval) == 0:
    print("⚠️  No stock-out events in 11/26–11/30 — skipping hold-out MAE.")
    overall_mae = overall_medae = None
else:
    test_eval["pred"] = model.predict(test_eval[FEATURES])
    overall_mae   = mean_absolute_error(test_eval["days_to_oos"], test_eval["pred"])
    overall_medae = median_absolute_error(test_eval["days_to_oos"], test_eval["pred"])
    print(f"Hold-out MAE  : {overall_mae:.2f} days")
    print(f"Hold-out MedAE: {overall_medae:.2f} days")

    # per-item MAE distribution
    per_item = (
        test_eval.groupby("itemsku")
                 .apply(lambda g: np.mean(np.abs(g["days_to_oos"] - g["pred"])))
                 .rename("item_mae")
    )
    print(per_item.describe(percentiles=[0.5, 0.75, 0.9]))

# ---------- 6b. Feature Importance (after model.fit) ----------
# import matplotlib.pyplot as plt

# imp_df = pd.DataFrame({
#     "feature": model.feature_name_,
#     "gain":    model.booster_.feature_importance(importance_type="gain"),
#     "split":   model.booster_.feature_importance(importance_type="split")
# }).sort_values("gain", ascending=False)

# print(imp_df.head(20))   # Top 20

# ---------- 6c. Plotting ----------
# topN = 15
# imp_df.head(topN).plot.barh(
#     x="feature", y="gain", figsize=(6, 5), legend=False
# )
# plt.gca().invert_yaxis()
# plt.title("Top Feature Gain")
# plt.tight_layout()
# plt.show()

# ---------- 7. Save model ----------
model.booster_.save_model(str(MODEL_OUT))
print(f"Model saved to {MODEL_OUT.resolve()}")

# ---------- 8. Save params.yaml ----------
def to_python(obj):
    """Recursively convert numpy scalars / arrays to builtin Python types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()                    # -> float or int
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj                               # str, bool, None, etc.

PARAMS_PATH = Path("models/prod_v1/params.yaml")
PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

params_dict = {
    "generated_on": str(pd.Timestamp.today()),
    "framework":    "LightGBM 4.3",
    "hyperparams":  lgbm_params,
    "categorical_features": CAT_FEATS,
    "numeric_features":     num_feats,
    "cv_mae_mean":  float(cv_mae.mean()),
    "cv_mae_std":   float(cv_mae.std()),
    "holdout_mae":  None if overall_mae is None else float(overall_mae),
    "holdout_medae": None if overall_medae is None else float(overall_medae),
    "training_rows": int(len(train_df)),
    "training_period": "2024-11-01 → 2024-11-25"
}

with open(PARAMS_PATH, "w") as f:
    yaml.safe_dump(to_python(params_dict), f, sort_keys=False)

print(f"Training parameters saved → {PARAMS_PATH}")

# ---------- 8. Inference helper ----------
def predict_days_to_oos(sample_df):
    """
    Input  : DataFrame with same columns as `train_df`
    Output : Series of predicted days_to_oos
    Rule   : If itemsku in permanent_oos_list => 0
    """
    sample_df = sample_df.copy()
    sample_df["itemsku"] = sample_df["itemsku"].astype("category")
    pred = model.predict(sample_df[FEATURES])
    pred = np.maximum(pred, 0)  # no negative days
    pred = np.round(pred).astype(int)
    pred[sample_df["itemsku"].isin(permanent_oos_list)] = 0
    return pd.Series(pred, index=sample_df.index, name="pred_days_to_oos")