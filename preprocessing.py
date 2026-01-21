"""
preprocessing.py

Normalization and latent state aggregation utilities.

Implements label-free preprocessing and z-score normalization
for latent physiological state vectors.
"""

import numpy as np
import pandas as pd


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies z-score normalization independently to each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    df_norm : pd.DataFrame
        Z-score normalized dataframe.
    """
    return (df - df.mean()) / (df.std(ddof=0) + 1e-6)


def construct_latent_states(features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates intermediate features into latent physiological
    state families.

    Latent state vector:
        S = [L, R, A, E, C, M]

    Returns
    -------
    latent_states : pd.DataFrame
        Normalized latent physiological state representation.
    """

    latent_states = pd.DataFrame(index=features.index)

    # -------------------------------------------------
    # L: Load Pressure State
    # -------------------------------------------------
    latent_states["L"] = features[
        ["load_volume", "load_intensity", "load_variability", "composite_load"]
    ].mean(axis=1)

    # -------------------------------------------------
    # R: Recovery Capacity State
    # -------------------------------------------------
    latent_states["R"] = features[
        ["rest_hours", "sleep_quality", "nutrition_score"]
    ].mean(axis=1)

    # -------------------------------------------------
    # A: Autonomic Stability State
    # -------------------------------------------------
    latent_states["A"] = features[
        ["autonomic_balance"]
    ].mean(axis=1)

    # -------------------------------------------------
    # E: Endocrine Stress State
    # -------------------------------------------------
    latent_states["E"] = features[
        ["hormonal_ratio", "log_ck"]
    ].mean(axis=1)

    # -------------------------------------------------
    # C: Cognitive–Affective State
    # -------------------------------------------------
    latent_states["C"] = features[
        ["cognitive_fatigue"]
    ].mean(axis=1)

    # -------------------------------------------------
    # M: Memory and History State
    # -------------------------------------------------
    latent_states["M"] = features[
        ["injury_history", "injury_load_interaction"]
    ].mean(axis=1)

    # -------------------------------------------------
    # Normalization (label-free, per-state)
    # -------------------------------------------------
    latent_states_normalized = zscore_normalize(latent_states)

    return latent_states_normalized
