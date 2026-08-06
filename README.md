# PoC: Small-Data Creative Performance Modeling with Explainability

## Architecture 

![Architecture diagram](./architecture.png)

## Tech stack

| Category | Tools |
|---|---|
| Language | Python |
| Modeling | LightGBM |
| Explainability | SHAP |
| Data handling | Pandas, NumPy |
| Evaluation | Scikit-learn, SciPy |
| Visualization | Matplotlib |

This is a scoped proof-of-concept validating the core modeling decision from
my Section 3 answer: predicting + ranking video creative performance per
brand, under a small-per-brand-data constraint, with SHAP-based explanations.

**Scope note:** this does NOT call the Instagram API or process real video —
that's weeks of infrastructure work, out of scope for a same-day take-home.
Instead, `generate_data.py` synthesizes 200 videos (40/brand across 5 brands)
with realistic creative features (hook type, pacing, on-screen text timing,
audio energy, etc.) and a performance score with a known, brand-specific
ground-truth relationship baked in. This isolates and tests the *modeling
approach* — pooled global model + time-based split + SHAP — independent of
the (separately-solved) feature-extraction problem.

## Files
- `generate_data.py` — synthetic dataset generator
- `train_and_explain.py` — trains global LightGBM model (brand_id as feature),
  evaluates with a time-based split, and extracts SHAP explanations
- `synthetic_video_data.csv` — generated dataset

## Results

- **Test MAE: 11.46** (on a 0-100ish performance score scale)
- **Overall Spearman rank correlation: 0.38**
- **Per-brand ranking quality varies significantly**: brand_B rho=0.70,
  brand_E rho=0.54, brand_A rho=0.14, brand_C rho=0.20, brand_D rho=-0.18

## Honest finding: this surfaces a real limitation, not just a demo

The per-brand rank correlation swings from 0.70 to -0.18 across brands with
*identical* sample sizes (n=10 test videos each) — this is a direct,
reproducible demonstration of why the problem statement specifically flags
"small per-brand datasets" as a hard part of the system. It isn't a solved
problem just because you use a global model.

More specifically: `hook_type` and `brand_id` show **~0 SHAP importance** in
this run, despite brand-specific hook preference being explicitly baked into
the synthetic ground truth. With only ~30 training rows per brand, a pooled
tree model with `min_child_samples=5` doesn't reliably find the brand x hook
interaction — it's diluted by higher-volume continuous features like
`onscreen_text_early` and `video_len_sec`.

**What I'd change in a production version, based on this finding:**
1. Explicitly engineer a `hook_matches_brand_historical_preference` interaction
   feature rather than relying on the tree to discover brand x hook_type
   interactions from raw categoricals with this little data.
2. Move from single pooled GBM toward a hierarchical/partial-pooling model
   (e.g., a mixed-effects model or Bayesian hierarchical regression) so brand-
   level effects get explicit shrinkage instead of competing for tree splits.
3. Increase `min_child_samples` regularization search and check whether
   brand_id needs to be passed as a native categorical with higher priority,
   e.g., via monotonic constraints or by training small per-brand residual
   correctors on top of the global model's base prediction.

This is exactly the kind of thing that only shows up when you actually run
the pipeline end-to-end on data instead of just diagramming it on paper —
which is the point of this PoC.

## Ablation study

As a rigorous follow-up to the SHAP finding above, `ablation_study.py`
retrains the model multiple times, each time removing one feature group, to
independently test whether `hook_type` and `brand_id` genuinely contribute
nothing — or whether SHAP was just failing to surface real signal.

| Configuration | MAE | Δ MAE vs baseline | Spearman rho | Δ rho vs baseline |
|---|---|---|---|---|
| Full model (baseline) | 11.46 | — | 0.382 | — |
| Without brand_id | 11.46 | +0.00 | 0.382 | +0.000 |
| Without hook_type | 11.46 | +0.00 | 0.382 | +0.000 |
| Without pacing | 11.16 | -0.30 | 0.314 | -0.068 |
| Without audio_energy | 11.48 | +0.02 | 0.239 | -0.143 |
| Without on-screen text | 11.95 | +0.49 | 0.107 | -0.275 |
| Without visual_style_cluster | 11.74 | +0.28 | 0.294 | -0.088 |
| Without video_len_sec | 11.92 | +0.46 | 0.364 | -0.018 |

**Result: removing `brand_id` or `hook_type` changes literally nothing** —
identical MAE and rho to four decimal places. This independently confirms
the SHAP finding via a completely different method (direct ablation rather
than attribution), and rules out the possibility that SHAP was simply
failing to detect real signal.

By contrast, removing on-screen text causes the largest single drop in
ranking quality (rho falls from 0.38 to 0.11), followed by audio energy —
these are the features the model is actually relying on, and match the
production fix proposed above (engineering an explicit `hook_matches_brand`
interaction feature rather than trusting the model to discover it from raw
categoricals with this little data).

## Model explainability (visualizations)

Running `train_and_explain.py` saves two plots, which make the finding
above directly visible rather than just described in text.

### SHAP summary (beeswarm) plot
![SHAP summary](./shap_summary.png)

Each row is a feature; each dot is one test-set video. Dot position shows how
much that feature pushed the prediction up or down for that video, and color
shows whether the feature's value was high (pink) or low (blue). `hook_type`
and `brand_id` collapse to a flat gray line at zero — visual confirmation
that the model isn't using them at all, consistent with the small-per-brand-
data limitation discussed above.

### LightGBM feature importance (gain-based)
![Feature importance](./feature_importance.png)

A second, independent importance measure (based on how much each feature
improves split quality across the trees). Notably, `brand_id` and
`hook_type` don't appear on this chart at all — they contributed zero gain,
cross-confirming the SHAP finding via a completely different method.

Together, these plots are why this PoC is useful beyond a demo: they make it
easy to inspect what the model actually learned versus what it was supposed
to learn — exactly the kind of debugging visibility a real production ML
system needs.

## Quick start

**Requirements:** Python 3.12 (tested on 3.12.3). All dependencies are
pinned in `requirements.txt` for reproducible installs.

```bash
git clone https://github.com/palakchiraniya03/video-creative-performance-poc.git
cd video-creative-performance-poc
pip install -r requirements.txt

python3 generate_data.py       # generates synthetic_video_data.csv
python3 train_and_explain.py   # trains model, prints metrics, saves plots
python3 ablation_study.py      # runs the ablation study (optional)
```

Random seeds are fixed throughout (`np.random.seed(42)` for data generation,
`random_state=42` on the LightGBM model), so re-running produces identical
results: Test MAE 11.46, Spearman rho 0.382, on every run.

## Future work

- Instagram Graph API ingestion (replacing the synthetic data generator)
- CLIP embeddings for visual feature extraction
- Whisper transcription for audio/hook detection
- OCR for on-screen text extraction
- Bayesian hierarchical model for better per-brand pooling (see "Honest finding" above)
- Online/incremental learning as new performance data arrives
- Agentic creative recommendation layer (brief generation, budget reallocation, automated experimentation)
