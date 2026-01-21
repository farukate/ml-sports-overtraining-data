"""
feature_construction.py

Raw data processing and intermediate feature construction
for latent physiological state modeling.

This module implements interaction-based and ratio-based
descriptors used as building blocks for latent state aggregation.
"""

import pandas as pd
import numpy as np


def construct_intermediate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs physiologically meaningful intermediate features
    from raw athlete monitoring data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe loaded from the Kaggle dataset.

    Returns
    -------
    features : pd.DataFrame
        Dataframe containing intermediate descriptors used for
        latent state construction.
    """

    features = pd.DataFrame(index=df.index)

    # -------------------------------------------------
    # Load pressure proxies
    # -------------------------------------------------
    features["load_volume"] = df["TrainingHoursPerWeek"]
    features["load_intensity"] = df["TrainingIntensity"]
    features["load_variability"] = df["TrainingLoadVariation"]

    # Composite load proxy
    features["composite_load"] = (
        df["TrainingHoursPerWeek"] * df["TrainingIntensity"]
    )

    # -------------------------------------------------
    # Recovery-related descriptors
    # -------------------------------------------------
    features["rest_hours"] = df["RestHoursPerWeek"]
    features["sleep_quality"] = df["SleepQualityScore"]
    features["nutrition_score"] = df["NutritionScore"]

    # Load-to-recovery ratio (stress dominance)
    features["load_recovery_ratio"] = (
        features["composite_load"]
        / (df["RestHoursPerWeek"] + 1e-6)
    )

    # -------------------------------------------------
    # Autonomic regulation descriptors
    # -------------------------------------------------
    features["hr_rest"] = df["HeartRateRest"]
    features["hrv"] = df["HeartRateVar"]

    # Relative autonomic balance
    features["autonomic_balance"] = (
        df["HeartRateVar"] / (df["HeartRateRest"] + 1e-6)
    )

    # -------------------------------------------------
    # Endocrine and muscle stress descriptors
    # -------------------------------------------------
    features["cortisol"] = df["CortisolLevel"]
    features["testosterone"] = df["TestosteroneLevel"]
    features["ck_level"] = df["CKLevel"]

    # Catabolic–anabolic balance
    features["hormonal_ratio"] = (
        df["CortisolLevel"] / (df["TestosteroneLevel"] + 1e-6)
    )

    # Nonlinear muscle damage proxy
    features["log_ck"] = np.log1p(df["CKLevel"])

    # -------------------------------------------------
    # Cognitive–affective descriptors
    # -------------------------------------------------
    features["reaction_time"] = df["ReactionTime"]
    features["mood_score"] = df["MoodScore"]

    features["cognitive_fatigue"] = (
        df["ReactionTime"] / (df["MoodScore"] + 1e-6)
    )

    # -------------------------------------------------
    # Injury history interactions (memory effects)
    # -------------------------------------------------
    features["injury_history"] = df["InjuryHistory"]
    features["injury_load_interaction"] = (
        df["InjuryHistory"] * features["composite_load"]
    )

    return features
