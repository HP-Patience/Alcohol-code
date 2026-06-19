# Alcohol-code

Analysis code for the paper: *"Neuro-autonomic decoupling facilitates complex skill acquisition under mild disinhibition"*

This repository contains all analysis code used in the study. Raw data (eye tracking, GSR, EEG, PPG) are not included due to size constraints and participant confidentiality.

## Repository structure

```
Correlation/    — Cross-modality correlation heatmaps (EEG × GSR × eye × fly score)
EEG/            — EEG preprocessing, LMM, power band analysis, DE (dynamical entropy)
EYE/            — Eye-tracking pipeline: AOI classification, interpolation, LMM, missing data analysis
GSR/            — GSR/SCL signal processing, decomposition (SCL/SCR), LMM
Graph/          — Integrated visualizations across modalities (EEG, GSR, PPG, fly score)
ML-FlyScore/    — Elastic Net regression predicting flight score from physiological + neural features
Subject/        — Demographic information
scripts/        — Cross-modal analysis scripts (revision-related: no-session Elastic Net, FDR correction, random-slope LMM)
```

## Notebook naming convention

- Numbered notebooks (1, 2, 3...) should be run sequentially within each directory
- Suffix `b` indicates a variant for sensitivity analysis (e.g., `5b` = no-session version of Elastic Net)

## Software requirements

- Python 3 + Jupyter Notebook
- Key packages: pandas, numpy, scipy, statsmodels, scikit-learn
- EEG preprocessing: Brainstorm (MATLAB), GRETNA, BrainNet Viewer

## License

CC0 1.0 Universal
