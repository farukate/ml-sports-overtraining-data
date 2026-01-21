import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ======================================================
# AYARLAR
# ======================================================
LATENT_CSV = "athlete_overtraining_latent_states.csv"
RAW_CSV = "data.csv"   # Figür 3 için gerekli
FIG1_OUT = "Fig1_latent_pca1.png"
FIG2_OUT = "Fig2_latent_boundary2.png"
FIG3_OUT = "Fig3_raw_vs_latent3.png"


# ======================================================
# ORTAK FONKSİYON: PCA HAZIRLIK
# ======================================================
def compute_pca(X, n_components=2):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


# ======================================================
# FIGURE 1
# Latent Space – Unlabeled Geometry
# ======================================================
def figure_1_latent_unlabeled(df):
    X = df.drop(columns=["Overtraining"])
    X_pca, pca = compute_pca(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=18,
        alpha=0.4,
        color="gray"
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Latent Physiological State Space (Label-Free Embedding)")

    plt.tight_layout()
    plt.savefig(FIG1_OUT, dpi=300)
    plt.close()


# ======================================================
# FIGURE 2
# Boundary Emergence (Label Overlay)
# ======================================================
def figure_2_latent_with_labels(df):
    y = df["Overtraining"].values
    X = df.drop(columns=["Overtraining"])
    X_pca, pca = compute_pca(X)

    plt.figure(figsize=(6, 5))

    plt.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        s=18,
        alpha=0.4,
        label="Non-overtrained",
        color="#1f77b4"
    )

    plt.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        s=22,
        alpha=0.7,
        label="Overtrained",
        color="#d62728"
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Emergence of Boundary Structure Associated with Overtraining")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(FIG2_OUT, dpi=300)
    plt.close()


# ======================================================
# FIGURE 3
# Raw Feature Space vs Latent Space
# ======================================================
def figure_3_raw_vs_latent(raw_df, latent_df):
    # --- RAW
    y_raw = raw_df["Overtraining"].values
    X_raw = raw_df.drop(columns=["Overtraining"])
    X_raw_pca, _ = compute_pca(X_raw)

    # --- LATENT
    y_lat = latent_df["Overtraining"].values
    X_lat = latent_df.drop(columns=["Overtraining"])
    X_lat_pca, _ = compute_pca(X_lat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Raw
    axes[0].scatter(
        X_raw_pca[y_raw == 0, 0],
        X_raw_pca[y_raw == 0, 1],
        s=18,
        alpha=0.4,
        color="#1f77b4",
        label="Non-overtrained"
    )
    axes[0].scatter(
        X_raw_pca[y_raw == 1, 0],
        X_raw_pca[y_raw == 1, 1],
        s=22,
        alpha=0.7,
        color="#d62728",
        label="Overtrained"
    )
    axes[0].set_title("(a) Raw Feature Space (PCA Projection)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # (b) Latent
    axes[1].scatter(
        X_lat_pca[y_lat == 0, 0],
        X_lat_pca[y_lat == 0, 1],
        s=18,
        alpha=0.4,
        color="#1f77b4"
    )
    axes[1].scatter(
        X_lat_pca[y_lat == 1, 0],
        X_lat_pca[y_lat == 1, 1],
        s=22,
        alpha=0.7,
        color="#d62728"
    )
    axes[1].set_title("(b) Latent Physiological State Space (PCA Projection)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(FIG3_OUT, dpi=300)
    plt.close()


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("Loading data...")
    latent_df = pd.read_csv(LATENT_CSV)
    raw_df = pd.read_csv(RAW_CSV)

    print("Generating Figure 1...")
    figure_1_latent_unlabeled(latent_df)

    print("Generating Figure 2...")
    figure_2_latent_with_labels(latent_df)

    print("Generating Figure 3...")
    figure_3_raw_vs_latent(raw_df, latent_df)

    print("All figures generated successfully.")
