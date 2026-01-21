# Overtraining as a Boundary Phenomenon in Latent Physiological State Spaces

This repository contains the code used in the paper:

**Overtraining as a Boundary Phenomenon in Latent Physiological State Spaces**  
submitted to the *Journal of Machine Learning Research (JMLR)*.

## Overview
The code implements a physiologically grounded latent state representation for analyzing overtraining as a systemic and multivariate phenomenon. The focus is on representation construction and geometric analysis rather than predictive modeling.

## Dataset
The dataset used in this study is publicly available on Kaggle:

Athlete Overtraining Prediction Dataset  
https://www.kaggle.com/datasets/yuanchunhong/athlete-overtraining-prediction-dataset

Due to licensing and reproducibility considerations, the dataset is **not redistributed** in this repository.

## Repository Structure
- `src/feature_construction.py`  
  Raw data processing and intermediate feature construction.

- `src/latent_state_construction.py`  
  Aggregation of intermediate descriptors into latent physiological state families.

- `src/preprocessing.py`  
  Normalization and preprocessing steps applied to latent states.

- `src/visualization.py`  
  Code for generating Figures 1–3 in the manuscript.

- `figures/`  
  Generated figures corresponding to the Results section.

## Usage
1. Download the dataset from Kaggle.
2. Place the CSV file in a local directory.
3. Update the dataset path in the scripts if needed.
4. Run the scripts in the order described above.

## Dependencies
See `requirements.txt` for required Python packages.

## Reproducibility
All experiments are deterministic given the same input data and random seed settings.

## License
This project is released under the MIT License.
