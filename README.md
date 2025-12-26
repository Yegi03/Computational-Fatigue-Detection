# 📋 FINAL VERSION - Complete Package for Paper

This folder contains everything needed to reproduce the paper: **"Calibrated Multimodal Fatigue Detection from Physiological Signals"**

---

## 📁 **FOLDER STRUCTURE**

```
Final_version/
├── dataset/              # Complete ASCERTAIN dataset
│   ├── Dt_EEGFeatures.mat
│   ├── Dt_ECGFeatures.mat
│   ├── Dt_GSRFeatures.mat
│   ├── Dt_EMOFeatures.mat
│   ├── Dt_Personality.mat
│   ├── Dt_Order_Movie.mat
│   ├── Dt_SelfReports.mat
│   └── *.xlsx, *.xls files
│
├── code/                 # All code files
│   ├── run_rq1_pfi_accuracy.py        # RQ1 analysis
│   ├── run_rq2_temporal_ceiling.py   # RQ2 analysis
│   ├── run_rq3_modality_ranking.py    # RQ3 analysis
│   ├── run_ablation_study_simple.py   # Ablation study (LoRA-only, MoE-only, LoRA+MoE)
│   ├── run_all_analyses.py            # Master analysis script
│   ├── create_all_figures.py         # Generate F01-F04 figures
│   ├── create_additional_figures.py   # Generate F05-F07 figures
│   ├── create_f08_ablation_study.py   # Generate F08 ablation figure
│   ├── create_verified_tables.py      # Generate LaTeX tables
│   ├── enhanced_tiny_deep.py          # Main model (LoRA + MoE)
│   ├── lora.py                        # LoRA (Low-Rank Adaptation) implementation
│   ├── moe.py                         # MoE (Mixture of Experts) implementation
│   ├── tiny_deep.py                   # Baseline model
│   └── pfi_literature.py              # PFI calculation
│
├── run_complete_analysis_safe.py      # Complete analysis pipeline
├── run_simple.py                      # Simple version (tables only)
├── requirements.txt                   # Python dependencies
│
├── results/             # Generated results
│   ├── complete_analysis_summary.json
│   ├── rq1_pfi_accuracy.json
│   ├── rq2_temporal_ceiling.json
│   ├── rq3_modality_ranking.json
│   └── ablation_study_results.json
│
├── tables/              # Generated LaTeX tables
│   ├── T01_rq1_pfi_accuracy.tex
│   ├── T02_rq2_temporal_ceiling.tex
│   └── T03_rq3_modality_contributions.tex
│
├── figures/             # Generated figures (PDF + PNG)
│   ├── F01_rq1_pfi_accuracy.pdf/.png          # PFI accuracy (MAE, CCC, r, ECE)
│   ├── F02_rq2_temporal_ceiling.pdf/.png     # Temporal ceiling (ToT vs ACL)
│   ├── F03_rq3_modality_contributions.pdf/.png # Modality contributions
│   ├── F04_performance_comparison.pdf/.png    # PFI performance (MAE + CCC)
│   ├── F05_architecture_diagram.pdf/.png      # Model architecture
│   ├── F06_parameter_efficiency.pdf/.png      # Parameter efficiency
│   ├── F07_calibration_improvement.pdf/.png   # Calibration curves
│   └── F08_ablation_study.pdf/.png            # Ablation study (PFI CCC + MAE)
│
└── README.md            # This file
```

---

## 🚀 **QUICK START**

### **1. Install Dependencies**

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch
```

### **2. Run Complete Analysis**

```bash
cd Final_version
python run_complete_analysis_safe.py
```

**Alternative (simpler version):**
```bash
python run_simple.py  # Generates tables and results only
```

**Generate figures only:**
```bash
cd code
python -c "from create_all_figures import *; from create_additional_figures import *; from create_f08_ablation_study import *; from pathlib import Path; output_dir = Path('../figures'); create_f01_rq1_pfi_accuracy(output_dir); create_f02_rq2_temporal_ceiling(output_dir); create_f03_rq3_modality_contributions(output_dir); create_f04_performance_comparison(output_dir); create_f05_architecture_diagram(output_dir); create_f06_parameter_efficiency(output_dir); create_f07_calibration_improvement(output_dir); create_f08_ablation_study(output_dir)"
```

This single command will:
- ✅ Load ASCERTAIN dataset
- ✅ Run all 3 research question analyses
- ✅ Run ablation study (LoRA-only, MoE-only, LoRA+MoE)
- ✅ Generate all tables
- ✅ Generate all figures (F01-F08)
- ✅ Create final summary

---

## 📊 **WHAT THIS PACKAGE CONTAINS**

### **1. Dataset (Complete ASCERTAIN)**
- **58 subjects** × **36 clips** per subject
- **EEG**: 8 channels @ 32 Hz → 24 features per 10-second window
- **ECG/HRV**: 2 channels @ 256 Hz → 8 features per 60-second window
- **EDA/GSR**: 1 channel @ 128 Hz → 5 features per 10-second window
- **Self-reports**: Valence and Arousal per clip
- **Personality**: Big-Five traits per subject

### **2. Code (Complete Implementation)**

#### **Master Script**: `run_complete_analysis.py`
- Orchestrates entire pipeline
- Loads dataset
- Runs all analyses
- Generates all outputs

#### **Model Architecture**:
- **`enhanced_tiny_deep.py`**: Main multimodal model with LoRA and MoE support
- **`lora.py`**: LoRA (Low-Rank Adaptation) implementation (rank=16, α=16, dropout=0.05)
- **`moe.py`**: Mixture of Experts (MoE) implementation (K=3 experts, softmax gating)
- **`tiny_deep.py`**: Baseline Tiny-Deep model (full fine-tuning)
- **`pfi_literature.py`**: PFI target construction

#### **Analysis Scripts**:
- **`run_rq1_pfi_accuracy.py`**: PFI accuracy vs classical regressors
- **`run_rq2_temporal_ceiling.py`**: Temporal confounding analysis (ToT vs ACL)
- **`run_rq3_modality_ranking.py`**: Modality contributions ranking
- **`run_ablation_study_simple.py`**: Ablation study (Full FT, LoRA-only, MoE-only, LoRA+MoE)

#### **Output Generation**:
- **`create_all_figures.py`**: Generates F01-F04 (main results figures)
- **`create_additional_figures.py`**: Generates F05-F07 (methodology figures)
- **`create_f08_ablation_study.py`**: Generates F08 (ablation study figure)
- **`create_verified_tables.py`**: Generates all LaTeX tables (T01-T03)

---

## 📈 **RESULTS IN THIS PACKAGE**

### **RQ1: PFI Accuracy (Regression)**
- **Our method (LoRA+MoE)**: MAE 0.095, CCC 0.680, Pearson r 0.690
- **LoRA-only**: MAE 0.112, CCC 0.620
- **MoE-only**: MAE 0.105, CCC 0.635
- **Best classical (XGB-Reg)**: MAE 0.128, CCC 0.553
- **Full Fine-tuning**: MAE 0.118, CCC 0.587
- **Improvement**: 25.8% MAE reduction vs best classical, 19.5% vs full fine-tuning

### **RQ2: Temporal Ceiling (Classification)**
- **Time-only baseline**: F1 1.000 (deterministic ceiling)
- **Our method (LoRA+MoE)**: ToT F1 0.585, ACL F1 0.545
- **LoRA-only**: ToT F1 0.571, ACL F1 0.519
- **MoE-only**: ToT F1 0.560, ACL F1 0.510
- **Interpretation**: ToT is strongly confounded by temporal ordering; ACL shows physiology contribution

### **RQ3: Modality Contributions**
- **Single modalities**: ECG (CCC 0.587) > EEG (0.523) > GSR (0.498)
- **Best combination**: All modalities (CCC 0.680)
- **Synergy benefit**: +0.093 CCC over best single modality (ECG)

### **Ablation Study**
- **Full Fine-tuning**: 2.2M params, CCC 0.587, MAE 0.118
- **LoRA-only**: 50K params (2.3%), CCC 0.620, MAE 0.112
- **MoE-only**: 282K params (12.9%), CCC 0.635, MAE 0.105
- **LoRA+MoE**: 332K params (15.1%), CCC 0.680, MAE 0.095 ⭐ **Best**

---

## 🔧 **TECHNICAL DETAILS**

### **Model Architecture**
- **EEG Encoder**: 1D-CNN (2 layers, 32-64 filters) → AdaptiveAvgPool → LoRA Linear
- **ECG Encoder**: 2-layer GRU (hidden=64) → LoRA Linear
- **GSR Encoder**: 1-layer GRU (hidden=32) → LoRA Linear
- **Fusion**: Concatenation → LayerNorm → 256-d vector
- **MoE**: 3 expert networks (hidden=128, 2 layers each), softmax gating
- **LoRA**: rank=16, α=16, dropout=0.05
- **Task Heads**: PFI (regression, Linear), ToT/ACL (classification, Sigmoid)
- **Trainable parameters**: 
  - LoRA-only: 50K (2.3% of full model)
  - MoE-only: 282K (12.9% of full model)
  - LoRA+MoE: 332K (15.1% of full model)
  - Full Fine-tuning: 2.2M (100%)

### **Evaluation Protocol**
- **Method**: Nested Leave-One-Subject-Out (LOSO)
- **Outer folds**: 58 (one per subject)
- **Inner split**: 80/20 (46 fit, 11 calibration)
- **Test**: 1 subject per fold (untouched until evaluation)
- **Metrics**: Median [IQR] across subjects

### **Signal Processing**
- **EEG**: 10-second windows, 5-second stride
  - Band-pass: 1-45 Hz, notch: 50/60 Hz
  - Welch periodograms: 2-second Hamming, 50% overlap
  - Features: δ, θ, α, β band powers, ratios, LZC, Hjorth
- **HRV**: 60-second windows, 10-second stride
  - Pan-Tompkins R-peak detection
  - Lomb-Scargle spectral analysis
  - Features: SDNN, RMSSD, LF, HF, LF/HF
- **EDA**: 10-second windows, 5-second stride
  - Low-pass: 1 Hz, cvxEDA decomposition
  - Features: Tonic SCL, phasic SCRs

---

## 📝 **OUTPUT FILES**

### **Tables** (LaTeX format)
1. **T01_rq1_pfi_accuracy.tex**: PFI regression results (all methods including LoRA-only, MoE-only, LoRA+MoE)
2. **T02_rq2_temporal_ceiling.tex**: ToT and ACL classification results
3. **T03_rq3_modality_contributions.tex**: Modality ablation results

### **Figures** (PDF + PNG)
1. **F01_rq1_pfi_accuracy.pdf/.png**: PFI accuracy comparison (4-panel: MAE, CCC, Pearson r, ECE)
2. **F02_rq2_temporal_ceiling.pdf/.png**: Temporal ceiling analysis (ToT vs ACL side-by-side)
3. **F03_rq3_modality_contributions.pdf/.png**: Modality contributions (sorted by CCC)
4. **F04_performance_comparison.pdf/.png**: PFI performance comparison (MAE + CCC)
5. **F05_architecture_diagram.pdf/.png**: Model architecture diagram
6. **F06_parameter_efficiency.pdf/.png**: Parameter efficiency comparison
7. **F07_calibration_improvement.pdf/.png**: Calibration curves (before/after isotonic)
8. **F08_ablation_study.pdf/.png**: Ablation study (PFI CCC + MAE across variants)

### **Results** (JSON)
- **complete_analysis_summary.json**: Complete results summary
- **rq1_pfi_accuracy.json**: RQ1 detailed results
- **rq2_temporal_ceiling.json**: RQ2 detailed results
- **rq3_modality_ranking.json**: RQ3 detailed results
- **ablation_study_results.json**: Ablation study results

---

## ✅ **VERIFICATION CHECKLIST**

Before submission, verify:
- [ ] All dataset files present in `dataset/`
- [ ] All code files present in `code/`
- [ ] Master script runs without errors
- [ ] All tables generated correctly
- [ ] All figures generated correctly
- [ ] Results match paper claims
- [ ] Reproducibility confirmed (same seed = same results)

---

## 🔬 **REPRODUCIBILITY**

- **Fixed seed**: 1337 (set in all analysis scripts)
- **Deterministic execution**: Enabled (torch.manual_seed, numpy.random.seed)
- **One-command reproduction**: `python run_complete_analysis_safe.py`
- **Evaluation protocol**: Nested Leave-One-Subject-Out (LOSO) with 58 folds
- **All hyperparameters**: Documented in code comments
- **Model configurations**: 
  - LoRA: rank=16, alpha=16, dropout=0.05
  - MoE: K=3 experts, hidden_dim=128, num_layers=2, gate_type='soft', temperature=1.0

---


## 📧 **CONTACT**

For questions, issues, or collaboration inquiries, please contact us at yza5171@psu.edu or skr6024@psu.edu.

## 📄 **LICENSE**

See LICENSE file in parent directory.

---

**Last Updated**: 2024  
**Status**: ✅ **COMPLETE** - Ready for paper submission

