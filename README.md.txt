markdown

# MSGDE: Multi-Scale Graph Dynamical Evolution Framework for EEG Classification

## Overview

This repository provides the complete computational framework developed in our study for EEG-based epilepsy detection and classification. The framework integrates multi-scale graph filtering, nonlinear dynamical evolution, and machine learning, and has been validated on three publicly available EEG datasets: **Bonn**, **New Delhi**, and **CHB-MIT**.

The workflow consists of five major stages:

1. **Data preparation** (pre‑segmented samples or sliding‑window segmentation for continuous recordings).
2. **Sample‑wise similarity matrix construction**.
3. **Multi‑scale graph Laplacian generation**.
4. **Nonlinear dynamical evolution** on each graph scale.
5. **Feature extraction** and **classification** using both traditional ML and deep learning models.

---

# Workflow (Unified)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Three EEG Datasets                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Bonn          │   New Delhi     │   CHB-MIT                   │
│   (pre‑segmented)│  (pre‑segmented)│  (continuous recordings)    │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                        │
         │                 │                        ▼
         │                 │               Preprocessing & sliding‑window
         │                 │               (23 channels, 4 s, 1 s step)
         │                 │                        │
         └─────────────────┴────────────────────────┘
                                    │
                                    ▼
                    Sample‑wise Pearson similarity matrix (N×N)
                                    │
                                    ▼
                    Multi‑scale graph Laplacian family (K scales)
                                    │
                                    ▼
                    Coupled Lorenz dynamics on each scale
                                    │
                                    ▼
                    Dynamical trajectory extraction
                                    │
                                    ▼
                    Feature extraction
                    (Chaos + Synchronization descriptors)
                                    │
                                    ▼
                    Machine learning classification
                    (KNN, RF, GBDT, SVM, LSTM, RNN)
                                    │
                                    ▼
                    Performance evaluation & noise robustness test

1. Datasets

DatasetSubjectsChannelsSampling RateNatureNumber of ClassesSamplesBonnHealthy / Epileptic1173.61 HzPre‑segmented5500 (100 per class)New DelhiEpileptic4 (or as used)200 HzPre‑segmented3450 (150 per class)CHB-MIT23 pediatric patients23 bipolar256 HzContinuous recordings2 (ictal / interictal)Variable (sliding‑window)

2. Data Preparation

2.1 Bonn and New Delhi Datasets

These datasets consist of fixed‑length EEG epochs that are already pre‑segmented. No additional window extraction or channel selection is performed. Each epoch is used directly as a sample.

Bonn: 500 samples, each of length 4,096 points (≈23.6 s), single channel.

New Delhi: 450 samples, each of length 512 points (≈2.56 s), 4 or 16 channels (depending on version). For consistency, only the first 4 channels are used (or all available channels are flattened).

2.2 CHB‑MIT Dataset

Continuous scalp EEG recordings from 23 patients were preprocessed as follows:

Resampling to 256 Hz.

Band‑pass filtering (0.5–30 Hz, fourth‑order Butterworth, zero‑phase) and notch filtering (50 Hz, Q=30).

Baseline correction (subtract mean of first 0.2 s of each window).

Outlier replacement using median absolute deviation (MAD, threshold=8.0).

Trial‑wise standardization (zero mean, unit variance per channel).

Window segmentation: 4 s windows, sliding step 1 s (75% overlap).

Label assignment: ictal (overlapping annotated seizure intervals) vs. interictal (at least 10 s away from any seizure).

Balanced sampling: each ictal window paired with 2 interictal windows (1:2 ratio).

After preprocessing, each window is flattened into a 1D vector.

3. Similarity Matrix Construction (All Datasets)

For each dataset, the sample‑wise similarity matrix is constructed identically:

Each sample (pre‑segmented epoch or extracted window) is flattened into a 1D vector.

Vectors are mean‑centered and L2‑normalized.

Pearson correlation is computed between every pair of vectors.

Negative correlations are clipped to zero, and self‑connections (diagonal) are removed.

Result: a non‑negative sample‑wise similarity matrix S ∈ ℝ<sup>N×N</sup>.

4. Multi‑Scale Graph Laplacian Generation

The similarity matrix S is treated as a weighted graph.

Edge weights are the similarity values.

The edge‑weight distribution is uniformly partitioned into K = 10 intervals.

For each interval, a subgraph is constructed by retaining only edges whose weights fall into that interval.

For each subgraph, the corresponding graph Laplacian is computed.

Result: a family of K multi‑scale graph Laplacians.

5. Nonlinear Dynamical Evolution

5.1 Coupled Lorenz System

Each graph node is assigned a Lorenz oscillator:

math

\dot{x} = σ (y − x)
\dot{y} = x (ρ − z) − y
\dot{z} = x y − β z

with σ = 10, β = 8/3, ρ = 60.

Nodes are coupled via the graph Laplacian (diffusive coupling) using the xyz_all coupling mode.

5.2 Numerical Integration

Solver: fourth‑order Runge–Kutta (RK4)

Time step: 1×10⁻³

Total iterations: 3,000

Transient iterations: 2,000 (discarded)

The coupling strength ε is varied from 0 to 20 with step 0.5 in ablation studies.

5.3 Sampling and Initial Conditions

Sampling strategy: hybrid (random + uniform)

Random ratio: 0.3

Initial states: x ∈ [-10,10], y ∈ [-10,10], z ∈ [0,50]

5.4 Trajectory Output

For each scale, each node, and each coupling strength, the final 1,000 time steps (after transients) are saved as trajectories.npy (shape: K × N_nodes × 1000 × 3).

6. Feature Extraction

From each node’s trajectory, two types of features are extracted:

Chaos features: maximum Lyapunov exponent (estimates system complexity)

Synchronization features: correlation between the node’s trajectory and a reference node

These descriptors are concatenated into a feature vector for each node.

For each scale and each coupling strength, a feature matrix of shape N_nodes × n_features is saved as features_scale_XX.npy.

7. Machine Learning Classification

7.1 Classifiers Evaluated

Traditional ML: K‑Nearest Neighbors (KNN), Random Forest (RF), Gradient Boosting Decision Trees (GBDT), Support Vector Machine (SVM)

Deep Learning: Long Short‑Term Memory (LSTM), Recurrent Neural Network (RNN)

7.2 Input Features

Dynamical features (chaos + synchronization) extracted from the trajectories.

For each experiment, the best‑performing scale and coupling strength are selected via cross‑validation.

7.3 Evaluation Protocol

Bonn and New Delhi: 10‑fold stratified cross‑validation (with predefined classification cases).

CHB‑MIT: Patient‑wise cross‑validation to prevent data leakage across subjects.

Metrics: Accuracy, Precision, Sensitivity, Specificity.

8. Noise Robustness Evaluation

To assess the framework’s resilience to noise (applied to CHB‑MIT only, due to its clinical nature):

Gaussian white noise is added to the clean EEG windows at various signal‑to‑noise levels (noise ratios from 0.1 to 1.0, step 0.1).

The entire pipeline is repeated for each noise level.

Performance degradation is measured to quantify robustness.

Software Requirements

Python 3.8+

NumPy

SciPy

scikit‑learn

mne (for CHB‑MIT EDF reading)

PyTorch (for deep learning models)

Matplotlib

seaborn

pandas

PyYAML

rich (for logging)

Directory Structure

text

case21/
│
├── config/
│   └── config_chbmit.yaml          # Main configuration file (applies to all datasets)
│
├── src/
│   ├── data/
│   │   └── chbmit_loader.py        # CHB‑MIT data loader (summary parsing, EDF reading)
│   ├── preprocessing/
│   │   ├── filters.py              # Band‑pass and notch filtering
│   │   ├── baseline.py             # Baseline correction
│   │   ├── outliers.py             # MAD‑based outlier replacement
│   │   ├── standardize.py          # Trial‑wise standardization
│   │   └── pipeline_chbmit.py      # Full preprocessing pipeline for CHB‑MIT
│   ├── laplacian/                  # Multi‑scale Laplacian generation (external)
│   ├── dynamics/                   # Lorenz dynamics and trajectory generation (external)
│   └── utils/                      # Logging, timing, I/O utilities
│
├── run_preprocess_chbmit.py        # Preprocessing + similarity matrix for CHB‑MIT
├── run_preprocess_noise.py         # Noise robustness script (CHB‑MIT)
├── multiscale_filter.py            # Multi‑scale Laplacian generation (all datasets)
├── dynamics_evolution.py           # Dynamical evolution per coupling strength
├── train.py                        # Machine learning training and evaluation
├── submit_all_job_gpu_cpu.sh       # SLURM submission script for parallel jobs
│
└── README.md

Output

The workflow generates:

Sample‑wise similarity matrices (similarity_matrix.npy) and labels (labels.npy) for each dataset.

Multi‑scale graph Laplacian families (laplacian_family_K10_sigma3.0.npy).

Dynamical trajectories (trajectories.npy) for each coupling strength.

Extracted dynamical features (features_scale_XX.npy).

Classification results (accuracy, confusion matrices, classification reports).

Noise robustness results (similarity matrices at multiple noise levels).

Preprocessing logs and statistics.
```