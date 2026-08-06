"""
Ablation study: retrains the same LightGBM model multiple times, each time
removing one feature GROUP, to test how much each group actually contributes
to predictive performance. This is a rigorous follow-up to the SHAP finding
(hook_type / brand_id showing ~0 importance) -- if removing a group barely
changes MAE/Spearman, that's independent confirmation it wasn't contributing.

Does NOT modify train_and_explain.py or the existing pipeline -- this is a
standalone additional experiment, run separately.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

df = pd.read_csv("/home/claude/synthetic_video_data.csv")
df["hook_type"] = df["hook_type"].astype("category")
df["brand_id"] = df["brand_id"].astype("category")

ALL_FEATURES = [
    "brand_id", "hook_type", "pacing_cuts_per_10s", "has_onscreen_text",
    "onscreen_text_early", "video_len_sec", "audio_energy", "visual_style_cluster",
]
target_col = "performance_score"

# ---- Feature groups to ablate (remove one group at a time) ----
FEATURE_GROUPS = {
    "Full model (baseline)": [],
    "Without brand_id": ["brand_id"],
    "Without hook_type": ["hook_type"],
    "Without pacing (pacing_cuts_per_10s)": ["pacing_cuts_per_10s"],
    "Without audio (audio_energy)": ["audio_energy"],
    "Without on-screen text (has_onscreen_text, onscreen_text_early)": ["has_onscreen_text", "onscreen_text_early"],
    "Without visual_style_cluster": ["visual_style_cluster"],
    "Without video_len_sec": ["video_len_sec"],
}

def time_based_split(data):
    train_frames, test_frames = [], []
    for brand, group in data.groupby("brand_id", observed=True):
        group = group.sort_values("post_order")
        split_idx = int(len(group) * 0.75)
        train_frames.append(group.iloc[:split_idx])
        test_frames.append(group.iloc[split_idx:])
    return pd.concat(train_frames), pd.concat(test_frames)

train_df, test_df = time_based_split(df)

results = []
for label, features_to_remove in FEATURE_GROUPS.items():
    feature_cols = [f for f in ALL_FEATURES if f not in features_to_remove]
    cat_features = [f for f in ["brand_id", "hook_type"] if f in feature_cols]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_samples=5, random_state=42, verbose=-1,
    )
    if cat_features:
        model.fit(X_train, y_train, categorical_feature=cat_features)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rho, _ = spearmanr(y_test, preds)

    results.append({"Configuration": label, "MAE": round(mae, 2), "Spearman rho": round(rho, 3)})
    print(f"{label:65s} MAE={mae:6.2f}  rho={rho:6.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/claude/ablation_results.csv", index=False)

# ---- Print as markdown table for direct README paste ----
print("\n--- Markdown table ---\n")
baseline_mae = results_df.iloc[0]["MAE"]
baseline_rho = results_df.iloc[0]["Spearman rho"]
print("| Configuration | MAE | Δ MAE vs baseline | Spearman rho | Δ rho vs baseline |")
print("|---|---|---|---|---|")
for _, row in results_df.iterrows():
    d_mae = round(row["MAE"] - baseline_mae, 2)
    d_rho = round(row["Spearman rho"] - baseline_rho, 3)
    d_mae_str = f"{d_mae:+.2f}" if row["Configuration"] != "Full model (baseline)" else "—"
    d_rho_str = f"{d_rho:+.3f}" if row["Configuration"] != "Full model (baseline)" else "—"
    print(f"| {row['Configuration']} | {row['MAE']} | {d_mae_str} | {row['Spearman rho']} | {d_rho_str} |")
