
Perfect bhai. Ab `config.yaml` aur `requirements.txt` bhi current project ke saath match kar rahe hain.

Ek important correction: README me **SMOTE/resampling wali line definitely hataani hai**, kyunki current requirements me `imbalanced-learn` bhi nahi hai.

Ab README ko proper current architecture ke according update karte hain.

### Next step — README replace karo

PowerShell me ye **exact command** paste karo:

````powershell
@'
# Credit Risk Engine

A credit risk assessment prototype built with Python, XGBoost, SHAP explainability, and Streamlit.

> **Note:** This is a machine-learning prototype for experimentation and demonstration. The configured risk thresholds and recommendations are demo policy rules and should not be treated as banking or lending standards.

## Architecture

```text
Customer Application Data
          |
          v
Data Validation
          |
          v
Feature Engineering
(DTI, EMI/Income, Loan/Income, Age, Employment, etc.)
          |
          v
Preprocessing
(Imputation + Scaling + One-Hot Encoding)
          |
          v
XGBoost Model
          |
          v
Probability of Default
          |
          v
Policy Engine
(Risk Score + Risk Band + Recommendation)
          |
          v
SHAP Explainability
          |
          v
Streamlit Dashboard
````

## Project Structure

```text
credit-risk-engine/
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── application_train.csv
│   └── processed/
│
├── models/
│   ├── xgboost_model.json
│   ├── preprocessing.pkl
│   ├── feature_names.pkl
│   └── evaluation/
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── score_distribution.png
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── validate_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   └── decision/
│       └── policy_engine.py
│
├── tests/
│   └── test_policy.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

## Features

The model uses the following core application features.

### Numerical

* `AMT_INCOME_TOTAL`
* `AMT_CREDIT`
* `AMT_ANNUITY`
* `DAYS_BIRTH`
* `DAYS_EMPLOYED`
* `CNT_FAM_MEMBERS`

### Categorical

* `CODE_GENDER`
* `FLAG_OWN_CAR`
* `FLAG_OWN_REALTY`
* `NAME_CONTRACT_TYPE`
* `NAME_EDUCATION_TYPE`
* `NAME_FAMILY_STATUS`
* `NAME_HOUSING_TYPE`
* `OCCUPATION_TYPE`

Feature engineering additionally creates financial and demographic derived features such as:

* Monthly income
* Debt-to-Income ratio
* Loan-to-Income ratio
* EMI-to-Income ratio
* Credit term
* Age in years
* Employment years
* Income per family member
* External credit-source average when available

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate on Windows

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Data

This project uses the Home Credit Default Risk dataset.

Place the training file at:

```text
data/raw/application_train.csv
```

The raw dataset is intentionally excluded from Git through `.gitignore` because of its large size.

## Training

Run:

```bash
python -m src.models.train
```

The training pipeline:

1. Loads the raw dataset.
2. Validates the required data.
3. Creates engineered financial features.
4. Selects the configured numerical and categorical features.
5. Splits the data into training and test sets.
6. Fits the preprocessing pipeline on training data.
7. Trains and compares:

   * Logistic Regression
   * Random Forest
   * XGBoost
8. Evaluates ROC-AUC, PR-AUC, precision, recall, and F1.
9. Saves the selected XGBoost model and preprocessing artifacts.

## Model Artifacts

Training creates the following artifacts:

```text
models/
├── xgboost_model.json
├── preprocessing.pkl
└── feature_names.pkl
```

These artifacts are used by the prediction and Streamlit application.

## Evaluation

Run evaluation using:

```powershell
python -c "from src.models.evaluate import evaluate_from_artifacts; print(evaluate_from_artifacts())"
```

Evaluation outputs are stored in:

```text
models/evaluation/
```

including:

* Classification report
* Confusion matrix
* ROC curve
* Score distribution

## Prediction

Single-application prediction can be performed through `predict_single()` in:

```text
src/models/predict.py
```

The prediction pipeline:

```text
Input Data
    ↓
Feature Engineering
    ↓
Raw Feature Alignment
    ↓
Saved Preprocessor
    ↓
XGBoost
    ↓
Probability of Default
    ↓
Policy Engine
    ↓
Risk Score / Band / Recommendation
```

## Risk Policy

The policy engine converts the model's probability of default into a risk band and recommendation.

Configured thresholds are:

```yaml
low_risk_threshold: 0.10
medium_risk_threshold: 0.25
```

The current policy produces:

* LOW risk
* MEDIUM risk
* HIGH risk

and an associated recommendation.

These thresholds are **demo configuration values**, not regulatory or banking standards.

## Explainability

SHAP is used to provide feature-level explanations for predictions.

The system can identify factors contributing to an individual prediction and expose them through the prediction/explainability pipeline.

## Streamlit Dashboard

Run:

```powershell
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides:

* Applicant input
* Default probability
* Risk score
* Risk band
* Recommendation
* SHAP-based explanation

## Testing

Run:

```powershell
pytest -q
```

The project includes unit tests for the policy engine.

## Configuration

Model and feature configuration is centralized in:

```text
config.yaml
```

This includes:

* Dataset paths
* Test split
* Random seed
* XGBoost class-weight ratio
* Risk thresholds
* Numerical features
* Categorical features

## Requirements

The project currently uses:

* Python 3.10+
* pandas
* NumPy
* scikit-learn
* XGBoost
* SHAP
* Streamlit
* joblib
* Matplotlib
* PyYAML
* pytest

Exact versions are pinned in `requirements.txt`.

## Current Model Performance

The current trained XGBoost model was evaluated on the project's test split with:

```text
ROC-AUC : 0.6721
PR-AUC  : 0.1578
Precision: 0.1329
Recall   : 0.5845
F1       : 0.2166
```

These metrics describe the current experimental model on this dataset split. They should not be interpreted as evidence of production lending performance.

## Important Limitations

Before using this system for real lending decisions, additional validation would be required, including:

* Probability calibration
* Out-of-time validation
* Robust threshold selection
* Data and feature leakage checks
* Subgroup/fairness analysis
* Stability monitoring
* Model drift monitoring
* Independent validation
* Regulatory and compliance review

## License / Usage

This project is intended for learning, experimentation, and demonstration of a credit-risk machine-learning pipeline.
'@ | Set-Content -Path .\README.md -Encoding UTF8

```

**Abhi koi aur command mat chalao.** Pehle ye command successfully execute ho jaye, phir mujhe batao.
```
