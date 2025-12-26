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
│   ├── run_rq1_pfi_accuracy.py     # RQ1 analysis
│   ├── run_rq2_temporal_ceiling.py # RQ2 analysis
│   ├── run_rq3_modality_ranking.py # RQ3 analysis
│   ├── create_verified_tables.py   # Table generation
│   ├── create_focused_figures.py   # Figure generation
│   ├── create_enhanced_performance_comparison.py
│   ├── create_rq3_modality_synergy_figure.py
│   ├── enhanced_tiny_deep.py        # Main model
│   ├── lora.py                      # LoRA implementation
│   ├── moe.py                       # MoE (Mixture of Experts) implementation
│   ├── tiny_deep.py                 # Baseline model
│   └── pfi_literature.py           # PFI calculation
│
├── run_complete_analysis_safe.py   # MASTER SCRIPT (runs everything)
├── run_simple.py                   # Simple version (tables only)
├── update_f03_pastel.py            # Update F03 figure with pastel colors
│
├── results/             # Generated results
│   └── complete_analysis_summary.json
│
├── tables/              # Generated LaTeX tables
│   ├── T01_rq1_pfi_accuracy.tex
│   ├── T02_rq2_temporal_ceiling.tex
│   └── T03_rq3_modality_contributions.tex
│
├── figures/             # Generated figures
│   ├── F01_rq1_pfi_accuracy.pdf
│   ├── F02_rq2_temporal_ceiling.pdf
│   ├── F03_rq3_modality_contributions.pdf
│   ├── F04_performance_comparison.pdf
│   └── F05_rq_summary.pdf
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

This single command will:
- ✅ Load ASCERTAIN dataset
- ✅ Run all 3 research question analyses
- ✅ Generate all tables
- ✅ Generate all figures
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
- **`enhanced_tiny_deep.py`**: Main multimodal model
- **`lora.py`**: LoRA (Low-Rank Adaptation) implementation
- **`moe.py`**: Mixture of Experts (MoE) implementation
- **`pfi_literature.py`**: PFI target construction

#### **Analysis Scripts**:
- **`run_rq1_pfi_accuracy.py`**: PFI accuracy vs classical regressors
- **`run_rq2_temporal_ceiling.py`**: Temporal confounding analysis
- **`run_rq3_modality_ranking.py`**: Modality contributions ranking

#### **Output Generation**:
- **`create_verified_tables.py`**: Generates all LaTeX tables
- **`create_focused_figures.py`**: Generates RQ1-RQ3 figures
- **`create_enhanced_performance_comparison.py`**: PFI performance comparison
- **`create_rq3_modality_synergy_figure.py`**: Modality synergy analysis

---

## 📈 **RESULTS IN THIS PACKAGE**

### **RQ1: PFI Accuracy**
- **Our method (post-cal)**: MAE 0.061, CCC 0.924, r 0.935
- **Best classical (XGB)**: MAE 0.128, CCC 0.553
- **Improvement**: 52.3% MAE reduction

### **RQ2: Temporal Ceiling**
- **Time-only baseline**: F1 1.000 (perfect ceiling)
- **Our method**: F1 0.578
- **Gap to ceiling**: 0.422
- **Interpretation**: ToT is strongly confounded by temporal ordering

### **RQ3: Modality Contributions**
- **Single modalities**: ECG (CCC 0.587) > EEG (0.523) > GSR (0.498)
- **Best combination**: All modalities (CCC 0.924)
- **Synergy benefit**: +0.337 CCC over best single modality

---

## 🔧 **TECHNICAL DETAILS**

### **Model Architecture**
- **EEG Encoder**: 1D-CNN (3 layers, 32-64-128 filters) with LoRA
- **ECG Encoder**: 2-layer GRU (hidden=64) with LoRA
- **GSR Encoder**: 1-layer GRU (hidden=32) with LoRA
- **Fusion**: Concatenation → LayerNorm → 256-d vector
- **LoRA**: rank=16, α=8, dropout=0.05
- **Trainable parameters**: ~50,000 (2.3% of full model)

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
1. **T01_rq1_pfi_accuracy.tex**: PFI regression results
2. **T02_rq2_temporal_ceiling.tex**: ToT classification results
3. **T03_rq3_modality_contributions.tex**: Modality ablation results

### **Figures** (PDF + PNG)
1. **F01_rq1_pfi_accuracy.pdf**: PFI accuracy comparison
2. **F02_rq2_temporal_ceiling.pdf**: Temporal ceiling analysis
3. **F03_rq3_modality_contributions.pdf**: Modality ranking
4. **F04_performance_comparison.pdf**: PFI performance (MAE + CCC)
5. **F05_rq_summary.pdf**: Summary of all 3 RQs

### **Results** (JSON)
- **complete_analysis_summary.json**: Complete results summary

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

- **Fixed seed**: 1337
- **Deterministic execution**: Enabled
- **One-command reproduction**: `python run_complete_analysis.py`
- **Data contracts**: Schema version 1.0.0
- **All hyperparameters**: Documented in code

---

## 📚 **CITATION**

If you use this code, please cite:

```bibtex
@article{your_paper_2024,
  title={Calibrated Multimodal Fatigue Detection from Physiological Signals: A Three-Question Analysis with Temporal Baseline Validation},
  author={Abdollahinejad, Yeganeh and Reza, Sayed Mohsin},
  journal={Your Journal},
  year={2024}
}
```

---

## 📧 **CONTACT**

For questions or issues:
- **Authors**: Yeganeh Abdollahinejad, Sayed Mohsin Reza
- **Institution**: Pennsylvania State University

---

## 📄 **LICENSE**

See LICENSE file in parent directory.

---

**Last Updated**: 2024  
**Status**: ✅ **COMPLETE** - Ready for paper submission

