# Non-Invasive Continuous Blood Glucose Estimation via Multi-Wavelength PPG Analysis

A production-grade Python machine learning platform designed for **non-invasive, continuous blood glucose estimation ($\text{mg/dL}$)** using multi-wavelength photoplethysmography (PPG) optical signals ($660\text{ nm}$, $730\text{ nm}$, $850\text{ nm}$, and $940\text{ nm}$).

---

## Key Highlights & System Architecture

- **Multi-Wavelength Optical Absorption**: Captures sub-surface hemodynamic shifts and vascular volume changes across $660\text{ nm}$, $730\text{ nm}$, $850\text{ nm}$, and $940\text{ nm}$.
- **Zero-Phase Digital Filtering**: 4th-order Butterworth bandpass filter ($0.5\text{ Hz} - 8.0\text{ Hz}$) eliminating baseline wander and high-frequency motion artifacts.
- **Biomedical & Morphological Feature Engine**:
  - Pulse Crest Time ($T_r$), Decay Time ($T_d$), Peak-to-Peak Interval ($PPI$).
  - Systolic vs Diastolic Inflection Ratios ($A_1/A_2$), $PW_{50}$, $PW_{75}$.
  - Velocity PPG ($VPG = \frac{dx}{dt}$) & Acceleration PPG ($APG = \frac{d^2x}{dt^2}$) $a,b,c,d,e$ wave arterial elasticity ratios.
  - Multi-wavelength optical density ratios ($R_{660/940}$, $R_{730/850}$, etc.).
- **Zero-Data-Leakage Validation**: GroupKFold cross-validation grouped strictly by Subject ID (`ID`).
- **Clinical Safety Evaluation**: Automated Clarke Error Grid (CEG) evaluation ensuring $>95\%$ predictions reside within Zone A (Clinically Accurate) and Zone B (Benign Errors).

```
 +-----------------------------------------------------------------------+
 |                         Hb_PPG_Dataset/                               |
 |   - subject information.xlsx (Demographics & Blood Glucose Target)    |
 |   - data_csv/ (252 Subject 4-Channel Synchronized PPG Files)          |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 | 1. Data Ingestion & Targeted Missing Value Imputation                 |
 |   - Coerces missing placeholders ('/') to NaN                         |
 |   - Target Conversion: mmol/L -> mg/dL                                |
 |   - Demographics: Median by Gender | Hb: KNN Imputer (k=5)            |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 | 2. Signal Preprocessing Engine (src/signal_processing.py)             |
 |   - 4th-Order Zero-Phase Butterworth Bandpass (0.5 - 8.0 Hz)          |
 |   - Per-Channel Z-Score Intensity Normalization                       |
 |   - Derivative Computation: VPG (dx/dt) & APG (d^2x/dt^2)             |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 | 3. Feature Extraction Engine (src/feature_extraction.py)              |
 |   - Pulse Morphology (Tr, Td, PPI, PW50, PW75, Area Ratios)           |
 |   - APG Elasticity Ratios (b/a, c/a, d/a, e/a, Aging Index)            |
 |   - Multi-Wavelength Optical AC/DC Density Ratios (R-values)          |
 |   --> Output: artifacts/cleaned_features.csv                          |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 | 4. ML Model Training & GroupKFold Cross Validation (src/model_trainer)|
 |   - Benchmark: RandomForest, ExtraTrees, LightGBM, XGBoost            |
 |   - RandomizedSearchCV Tuning on XGBoost Regressor                    |
 |   - Evaluation: MAE, RMSE, MARD (%)                                   |
 |   --> Output: artifacts/glucose_model.pkl                             |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------------------------------------------+
 | 5. Clinical Safety Evaluation (src/clinical_evaluator.py)              |
 |   - Clarke Error Grid Zone Breakdown (Zones A, B, C, D, E)            |
 |   - Clinical Safety Threshold: >95% in Zones A + B                    |
 |   --> Output: artifacts/clarke_error_grid.png                         |
 +-----------------------------------------------------------------------+
```

---

## Directory Structure

```plaintext
non_invasive_glucose_estimator/
├── .gitignore
├── README.md
├── requirements.txt
├── setup_env.sh
├── run_pipeline.py
├── Hb_PPG_Dataset/                  <-- Dataset Location
│   ├── subject information.xlsx
│   ├── data_csv/
│   │   ├── 1.csv
│   │   └── ... (252 Subject CSV Files)
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                    <-- Global constants, sampling rates, paths
│   ├── data_ingestion.py            <-- Excel metadata parsing & signal alignment
│   ├── signal_processing.py         <-- Butterworth bandpass filter & derivatives
│   ├── feature_extraction.py        <-- PPG morphology, APG ratios & AC/DC optical density
│   ├── model_trainer.py             <-- GroupKFold CV, XGBoost tuning, MARD metric
│   └── clinical_evaluator.py        <-- Clarke Error Grid computation & plot rendering
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py            <-- Data cleaning & imputation unit tests
│   ├── test_processing.py           <-- Digital filter & peak detection unit tests
│   └── test_model.py                <-- GroupKFold & Clarke Error Grid unit tests
└── artifacts/                        <-- Output models, clean feature matrices, and plots
    ├── cleaned_features.csv
    ├── glucose_model.pkl
    └── clarke_error_grid.png
```

---

## Quick Start & Installation

### 1. Environment Setup

Run the automated virtual environment initialization script:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Alternatively, set up manually:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pipeline Execution

To execute the entire end-to-end pipeline (Data Ingestion $\to$ Signal Preprocessing $\to$ Feature Extraction $\to$ Model Tuning $\to$ Clarke Error Grid Analysis):

```bash
python run_pipeline.py
```

### Running Unit Tests

Run the full pytest suite to verify system components:

```bash
pytest tests/ -v
```

---

## Expected Output Artifacts

Upon running `python run_pipeline.py`, the following files will be generated in `artifacts/`:

1. `artifacts/cleaned_features.csv`: Full tabular feature dataset containing subject demographics, pulse morphology metrics, derivative velocity/acceleration ratios, multi-wavelength optical density ratios, and target blood glucose values ($\text{mg/dL}$).
2. `artifacts/glucose_model.pkl`: Production binary of the hyperparameter-tuned XGBoost Regressor model.
3. `artifacts/clarke_error_grid.png`: High-resolution plot illustrating prediction distribution across Clarke Error Grid clinical zones (A, B, C, D, E).

---

## License & Citation

Dataset provided by:
*Chen et al., "A Four-Wavelength Photoplethysmogram dataset for non-invasive hemoglobin assessment", Figshare (2025).*
