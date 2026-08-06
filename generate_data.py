"""
Synthetic dataset generator for the Video Creative -> Performance PoC.

Simulates what a real feature-extraction pipeline (CLIP + Whisper + OCR +
shot-detection) would output per video, across multiple brands, over time.
This lets us validate the MODELING approach (small per-brand data, time-based
splits, explainability) without needing real Instagram API access.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

BRANDS = ["brand_A", "brand_B", "brand_C", "brand_D", "brand_E"]
HOOK_TYPES = ["question", "bold_claim", "pattern_interrupt", "product_first", "story"]

# Each brand has a slightly different "true" relationship between
# creative features and performance -- this is what makes per-brand
# modeling with shared global structure necessary.
BRAND_HOOK_PREFERENCE = {
    "brand_A": "question",
    "brand_B": "bold_claim",
    "brand_C": "pattern_interrupt",
    "brand_D": "product_first",
    "brand_E": "story",
}

def generate_videos(n_per_brand=40):
    rows = []
    video_id = 0
    for brand in BRANDS:
        # simulate videos posted over time (this matters for time-based splitting)
        for i in range(n_per_brand):
            hook = np.random.choice(HOOK_TYPES)
            pacing_cuts_per_10s = np.random.uniform(1, 8)
            has_onscreen_text = np.random.choice([0, 1])
            onscreen_text_early = np.random.choice([0, 1])  # within first 3s
            video_len_sec = np.random.uniform(8, 60)
            audio_energy = np.random.uniform(0, 1)
            visual_style_cluster = np.random.randint(0, 5)

            # ---- TRUE underlying signal (unknown to the model, this is ground truth) ----
            score = 50.0
            if hook == BRAND_HOOK_PREFERENCE[brand]:
                score += 25  # brand-specific hook affinity
            score += (onscreen_text_early * 10)
            score += (has_onscreen_text * 5)
            score -= abs(pacing_cuts_per_10s - 4) * 2  # sweet spot around 4 cuts/10s
            score += audio_energy * 8
            score -= (video_len_sec > 45) * 8  # long videos underperform slightly
            score += np.random.normal(0, 8)  # noise
            score = max(0, score)

            rows.append({
                "video_id": f"v{video_id}",
                "brand_id": brand,
                "post_order": i,  # proxy for time -- used for time-based split
                "hook_type": hook,
                "pacing_cuts_per_10s": round(pacing_cuts_per_10s, 2),
                "has_onscreen_text": has_onscreen_text,
                "onscreen_text_early": onscreen_text_early,
                "video_len_sec": round(video_len_sec, 1),
                "audio_energy": round(audio_energy, 2),
                "visual_style_cluster": visual_style_cluster,
                "performance_score": round(score, 2),  # target: composite of views/CTR/engagement
            })
            video_id += 1
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_videos(n_per_brand=40)
    df.to_csv("/home/claude/synthetic_video_data.csv", index=False)
    print(df.shape)
    print(df.head(10))
    print("\nPer-brand video counts:")
    print(df.brand_id.value_counts())
