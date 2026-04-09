<div align="center">

<a href="README.ru.md">🇷🇺 Читать на русском</a>

# 🛡️ FedGuard-IIoT

### Communication-Efficient and Robust Federated Anomaly Detection  
### for Resource-Constrained IIoT Edge Networks

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)](https://github.com/astral-sh/uv)
[![Typst](https://img.shields.io/badge/Typst-article-239DAD?style=for-the-badge)](https://typst.app)

*Sundetkhan Bekzat · Baibolat Bekarys*  
*Introduction to Multi-Agent Systems · Seema Rawat · 2026*

</div>

---

## 📌 About the Project

**FedGuard-IIoT** is a research project focused on **federated anomaly detection** in Industrial IoT (IIoT) edge networks. The goal is to design and experimentally validate a modular system that simultaneously solves three critical challenges in real-world edge deployments:

| Challenge | Our Solution |
|-----------|-------------|
| 🔗 **Limited uplink bandwidth** | Top-K sparsification + error-feedback |
| 📊 **Data heterogeneity (non-IID)** | FedProx + personalized thresholds |
| ☠️ **Byzantine / malicious clients** | Robust aggregation: RFA / Median / Krum |

> Unlike vanilla FedAvg that is vulnerable to attacks and instability under heterogeneous data, our system maintains high anomaly detection quality **even with 30% malicious clients** and **10× traffic reduction**.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IIoT Edge Clients                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Client 0 │  │ Client 1 │  │ Client 2 │  │ Client N │   │
│  │ (Line A) │  │ (SCADA)  │  │(Gateway) │  │  (PLC)   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│  [LSTM-AE Local Training] → [FedProx Loss] → [Top-K Compress]
└───────────────────────┬─────────────────────────────────────┘
                        │  Compressed Updates Δ̂ₖ
                ┌───────▼────────┐
                │   FL  Server   │
                │  ┌──────────┐  │
                │  │  RFA /   │  │
                │  │ Median / │  │  ← Robust Aggregation
                │  │  Krum    │  │
                │  └──────────┘  │
                │  Global θ      │
                └───────┬────────┘
                        │ Broadcast
                   [Edge Inference]
                  Anomaly Score > τ ?
```

---

## 🧠 ML Stack: Models & Methods

### 🔵 Model: LSTM Autoencoder (LSTM-AE)

The core anomaly detection model is a **reconstruction-based autoencoder** built on LSTM layers. Trained exclusively on **normal data** (fully unsupervised). An anomaly = high reconstruction error.

```
Input:  X ∈ ℝ^(W×F)   (window W=30 steps, F=6 sensors)
        ↓
  LSTM Encoder  →  z ∈ ℝ^16  (latent representation)
        ↓
  LSTM Decoder  →  X̂ ∈ ℝ^(W×F)
        ↓
Score:  s(X) = (1/WF) ‖X - X̂‖²_F    (anomaly score)
Anomaly if s(X) > τ   (τ = 95th percentile of training errors)
```

**Model specs:** Hidden `H=48` · Latent `L=16` · ~51K parameters · Suitable for edge hardware

---

### 🟢 Federated Learning Methods

| Method | Role | Key Parameter |
|--------|------|--------------|
| **FedAvg** | Baseline aggregator | — |
| **FedProx** | non-IID stabilization | μ = 0.01 (proximal term) |
| **Top-K sparsification** | Gradient compression | κ = 10% (top-10% values sent) |
| **Error-feedback** | Bias correction after compression | Accumulated residual eₖ |

Each round: sample 60% of clients → 3 local epochs → compress → aggregate → broadcast.

---

### 🔴 Robust Aggregation (Byzantine Defence)

| Aggregator | Principle | Robustness |
|------------|-----------|-----------|
| **FedAvg** | Arithmetic mean | ❌ Vulnerable |
| **Coord. Median** | Per-coordinate median | ✅ Good |
| **Trimmed Mean** | Trim 10% from each end | ✅ Good |
| **Krum** | Nearest-neighbour selection | ✅ Theoretically strict |
| **RFA** | Geometric median (Weiszfeld) | ✅ Best overall |

---

### ☠️ Adversarial Attack Scenarios

| Attack | Description |
|--------|-------------|
| **Model Poisoning** | Random/targeted noise injected into model updates |
| **Scaled Poisoning** | Gradient inversion + scaling to amplify damage |
| **Label Flipping** | Anomaly labels inverted during local training |
| **On-Off Attack** | Alternates honest and malicious rounds to evade detection |

---

## 📊 Experimental Results

### Communication Efficiency

| Configuration | AUROC | F1 | Traffic (MB) | Savings |
|---------------|-------|----|-------------|---------|
| FedAvg (baseline) | 0.703 | 0.231 | 22.37 | — |
| FedProx | 0.695 | 0.251 | 22.37 | — |
| **TopK-10%** | 0.690 | 0.248 | **2.24** | **10×** |
| **FedProx + TopK** | 0.701 | 0.229 | **2.24** | **10×** |

### Robustness Under Byzantine Attacks (model poisoning)

| Aggregator | α=0% | α=10% | α=20% | α=30% |
|------------|------|-------|-------|-------|
| FedAvg | 0.77 | 0.51 | 0.50 | 0.51 |
| **Median** | 0.77 | 0.72 | **0.82** | **0.78** |
| RFA | 0.64 | 0.66 | 0.71 | 0.70 |

### On-Off Evasion vs Persistent Attacks (α = 20%)

| | Persistent Poisoning | On-Off Attack |
|-|---------------------|---------------|
| FedAvg | 0.49 | 0.49 |
| **Median** | **0.84** | **0.82** |
| RFA | 0.71 | 0.68 |

---

## 🗂️ Project Structure

```
multi/
│
├── 📄 sn-article.typ          ← Typst source of the scientific article
├── 📄 sn-article.pdf          ← Compiled article (1.4 MB)
├── 📄 run_experiments.py      ← Main pipeline (runs everything)
├── 📄 pyproject.toml          ← uv dependencies
├── 🔧 typst                   ← Typst binary for PDF compilation
│
├── src/
│   ├── data_gen.py            ← Synthetic IIoT dataset generator
│   ├── models.py              ← LSTM Autoencoder + Deep SVDD
│   ├── fl_engine.py           ← FL engine (FedAvg/FedProx/TopK/aggregators)
│   ├── attacks.py             ← Attack implementations
│   ├── experiments.py         ← 5 experiments with metrics
│   └── visualization.py       ← 8 publication-quality figures
│
├── data/                      ← Auto-generated synthetic dataset (.npy)
├── figures/                   ← All 8 experiment figures (PNG)
├── results/
│   └── experiment_results.json
└── deep-research-report.md    ← Original research report
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run all experiments  (~26 min on GPU)
uv run python run_experiments.py

# 3. Compile the PDF article
./typst compile sn-article.typ sn-article.pdf
```

### Dependencies

```toml
torch >= 2.2.0        # Deep learning
numpy >= 1.26.0       # Numerics
scikit-learn >= 1.4.0 # Metrics (AUROC, F1)
matplotlib >= 3.8.0   # Plotting
seaborn >= 0.13.0     # Style
pandas >= 2.2.0       # Data handling
scipy >= 1.12.0       # Stats
tqdm >= 4.66.0        # Progress bars
```

---

## 📐 Dataset

Synthetic **IIoT multi-client benchmark** — 5 clients (edge nodes) with different parameters (non-IID by design):

| Client | Frequency | Amplitude | Noise | Role |
|--------|-----------|-----------|-------|------|
| 0 | 0.010 | 1.0 | 0.05 | Factory Line A |
| 1 | 0.018 | 1.4 | 0.08 | SCADA Node |
| 2 | 0.026 | 1.8 | 0.11 | Edge Gateway |
| 3 | 0.034 | 2.2 | 0.14 | Sensor Array |
| 4 | 0.042 | 2.6 | 0.17 | PLC Unit |

**Anomaly types injected:** Point spikes · Drift segments · Pattern shifts (~6% of data)

---

## 📖 Publication

The article is typeset in **Springer Nature** style using [Typst](https://typst.app):

```
Communication-Efficient and Robust Federated Anomaly Detection
for Resource-Constrained IIoT Edge Networks
under Non-IID and Adversarial Settings

Sundetkhan Bekzat, Baibolat Bekarys
Introduction to Multi-Agent Systems · Seema Rawat · 2026
```

---

<div align="center">

Made with ❤️ for **Introduction to Multi-Agent Systems**

</div>
