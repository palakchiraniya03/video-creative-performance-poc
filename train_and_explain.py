"""
Trains a global LightGBM model (pooled across brands, brand_id as a feature)
to predict video creative performance, using a TIME-BASED split per brand
(never random -- avoids leaking future creative trends into training).

Then uses SHAP to explain WHY specific videos are predicted to perform well,
which is the core "explainability" requirement from the problem statement.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

df = pd.read_csv("/home/claude/synthetic_video_data.csv")

# ---- Encode categoricals ----
df["hook_type"] = df["hook_type"].astype("category")
df["brand_id"] = df["brand_id"].astype("category")

feature_cols = [
    "brand_id", "hook_type", "pacing_cuts_per_10s", "has_onscreen_text",
    "onscreen_text_early", "video_len_sec", "audio_energy", "visual_style_cluster",
]
target_col = "performance_score"

# ---- TIME-BASED split per brand: train on first 75% posted, test on last 25% ----
train_frames, test_frames = [], []
for brand, group in df.groupby("brand_id", observed=True):
    group = group.sort_values("post_order")
    split_idx = int(len(group) * 0.75)
    train_frames.append(group.iloc[:split_idx])
    test_frames.append(group.iloc[split_idx:])

train_df = pd.concat(train_frames)
test_df = pd.concat(test_frames)

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# ---- Train global pooled model (LightGBM handles categoricals natively) ----
model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=5,  # small leaf size since per-brand data is limited
    verbose=-1,
)
model.fit(X_train, y_train, categorical_feature=["brand_id", "hook_type"])

# ---- Evaluate: MAE + ranking quality (Spearman) ----
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rho, _ = spearmanr(y_test, preds)

print(f"Test MAE: {mae:.2f}")
print(f"Spearman rank correlation (predicted vs actual ranking): {rho:.3f}")

# ---- Per-brand ranking quality (this is what the problem actually cares about) ----
print("\nPer-brand ranking quality (Spearman rho):")
test_df = test_df.copy()
test_df["predicted_score"] = preds
for brand, group in test_df.groupby("brand_id", observed=True):
    if len(group) > 2:
        rho_b, _ = spearmanr(group[target_col], group["predicted_score"])
        print(f"  {brand}: rho={rho_b:.3f}  (n={len(group)})")

# ---- SHAP explanations: WHY does a video rank where it does ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Pick brand_A's top-predicted test video and explain it
brand_a_test = test_df[test_df.brand_id == "brand_A"].copy()
brand_a_test["predicted_score"] = model.predict(brand_a_test[feature_cols])
top_video = brand_a_test.sort_values("predicted_score", ascending=False).iloc[0]
top_idx = brand_a_test["predicted_score"].idxmax()
row_position = test_df.index.get_loc(top_idx)

print(f"\n--- Explaining top-ranked predicted video for brand_A: {top_video['video_id']} ---")
print(f"Predicted score: {top_video['predicted_score']:.1f} | Actual: {top_video['performance_score']:.1f}")
shap_row = shap_values[row_position]
feature_importance = sorted(
    zip(feature_cols, shap_row), key=lambda x: abs(x[1]), reverse=True
)
print("Top feature contributions (SHAP values):")
for feat, val in feature_importance[:5]:
    direction = "+" if val > 0 else "-"
    print(f"  {feat:25s} {direction}{abs(val):.2f}  (feature value: {top_video[feat]})")

# Global feature importance across all predictions
print("\n--- Global feature importance (mean |SHAP|) ---")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
for feat, val in sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1]):
    print(f"  {feat:25s} {val:.2f}")
