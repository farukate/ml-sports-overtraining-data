# file: build_latent_states.py

import pandas as pd
import numpy as np

EPS = 1e-6

df = pd.read_csv("data.csv")

# --- Load Pressure (L)
df["L_load"] = df["TrainingHoursPerWeek"] * df["TrainingIntensity"]
df["L_variability"] = df["TrainingLoadVariation"]
df["L_load_recovery_ratio"] = df["L_load"] / (df["RestHoursPerWeek"] + EPS)

# --- Recovery Capacity (R)
df["R_recovery_capacity"] = df["RestHoursPerWeek"] * df["SleepQualityScore"]
df["R_sleep_load_ratio"] = df["SleepQualityScore"] / (df["L_load"] + EPS)
df["R_nutrition_sleep"] = df["NutritionScore"] * df["SleepQualityScore"]

# --- Autonomic Stability (A)
df["A_hrv_rest_ratio"] = df["HeartRateVar"] / (df["HeartRateRest"] + EPS)
df["A_autonomic_balance"] = df["HeartRateVar"] - df["HeartRateRest"]

# --- Endocrine Stress (E)
df["E_cort_test_ratio"] = df["CortisolLevel"] / (df["TestosteroneLevel"] + EPS)
df["E_ck_cortisol"] = df["CKLevel"] * df["CortisolLevel"]
df["E_log_ck"] = np.log(df["CKLevel"] + 1)

# --- Cognitive–Affective (C)
df["C_reaction_mood"] = df["ReactionTime"] / (df["MoodScore"] + EPS)
df["C_reaction_z"] = (df["ReactionTime"] - df["ReactionTime"].mean()) / df["ReactionTime"].std()

# --- Memory / History (M)
df["M_injury_flag"] = (df["InjuryHistory"] > 0).astype(int)
df["M_injury_load_interaction"] = df["InjuryHistory"] * df["L_load"]

latent_columns = [
    "L_load", "L_variability", "L_load_recovery_ratio",
    "R_recovery_capacity", "R_sleep_load_ratio", "R_nutrition_sleep",
    "A_hrv_rest_ratio", "A_autonomic_balance",
    "E_cort_test_ratio", "E_ck_cortisol", "E_log_ck",
    "C_reaction_mood", "C_reaction_z",
    "M_injury_flag", "M_injury_load_interaction",
    "Overtraining"
]

df_latent = df[latent_columns]
df_latent.to_csv("athlete_overtraining_latent_states.csv", index=False)

print("Latent state CSV created:", df_latent.shape)
