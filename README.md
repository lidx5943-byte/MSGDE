# MSGDE: Multi-Scale Graph Dynamical Evolution Framework for EEG Classification

This repository provides the complete implementation of the **MSGDE** framework for EEG-based epilepsy detection, integrating multi-scale graph construction, nonlinear dynamical modeling, and machine learning classification.

---

## 📁 Repository Structure

```
MSGDE/
├── code/                     # Core implementation
├── photo/                    # Figures
├── similarity/               # Precomputed similarity matrices
├── supporting_information/   # Supplementary materials
└── README.md
```

## 📊 Datasets

| Dataset | Subjects | Channels | Sampling Rate |
|:---|:---:|:---:|:---:|
| Bonn | 5 groups | 1 | 173.61 Hz |
| New Delhi | Epileptic | 1 | 200 Hz |
| CHB-MIT | 23 patients | 23 | 256 Hz |

## 📈 Key Results

| Dataset | Best Classifier | Accuracy |
|:---|:---|:---:|
| Bonn (16 cases) | RNN | **100.00%** |
| New Delhi (4 cases) | LSTM / RNN / KNN | **100.00%** |
| CHB-MIT (23 subjects) | GBDT | **99.64%** |

## 📝 Citation

If you use this code, please cite our paper:

```
@article{MSGDE2026,
  title={Multi-Scale Graph Dynamical Evolution for Robust EEG Classification},
  year={2026}
}
```

**Contact:** lidx5943@gmail.com
