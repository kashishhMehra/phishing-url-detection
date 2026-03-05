# Phishing URL Detector — XAI Framework

A two-layer machine learning framework for phishing URL detection with full explainability using SHAP and LIME. Built as part of a research paper exploring how Explainable AI (XAI) improves trust and transparency in cybersecurity models.

**Live Demo →** [url-phishing-detection-r.streamlit.app](https://url-phishing-detection-r.streamlit.app)

---

## Overview

Most phishing detection systems are black boxes — they give a result but no explanation. This project addresses that by combining ML classification with XAI explainability, allowing users to understand *why* a URL is flagged as phishing.

---

## Two-Layer Framework

**Layer 1 — Detection**
- Random Forest (primary classifier)
- Logistic Regression (baseline comparison)
- Trained on 549,346 URLs

**Layer 2 — Explainability**
- SHAP (SHapley Additive exPlanations) — global feature importance
- LIME (Local Interpretable Model-agnostic Explanations) — local per-prediction explanation

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — `phishing_site_urls.csv` |
| Total URLs | 549,346 |
| Legitimate | 392,924 (71.5%) |
| Phishing | 156,422 (28.5%) |
| Missing values | None |

---

## Features Extracted

16 features are extracted from raw URL strings:

| Feature | Description |
|---|---|
| `url_length` | Total character length of URL |
| `num_dots` | Number of `.` characters |
| `num_hyphens` | Number of `-` characters |
| `num_underscores` | Number of `_` characters |
| `num_slashes` | Number of `/` characters |
| `num_at` | Number of `@` characters |
| `num_question` | Number of `?` characters |
| `num_equal` | Number of `=` characters |
| `num_digits` | Count of digit characters |
| `has_https` | 1 if URL starts with `https` |
| `has_http` | 1 if URL starts with `http` |
| `has_ip` | 1 if URL contains an IP address |
| `num_subdomains` | Number of subdomains |
| `url_depth` | Depth based on `/` count |
| `has_suspicious_words` | 1 if URL contains phishing keywords |
| `is_trusted_domain` | 1 if domain is in known trusted list |

---

## Models

### Random Forest
- `n_estimators=50`, `max_depth=15`
- `class_weight='balanced'`
- Primary model used for final prediction

### Logistic Regression
- `C=1.0`, `max_iter=2000`
- `class_weight='balanced'`
- Baseline comparison model

Both models were trained on a **balanced dataset** (equal phishing and legitimate samples) to avoid class bias.

---

## Explainability

**SHAP** uses `LinearExplainer` on Logistic Regression to show which features globally pushed the prediction toward phishing or legitimate.

**LIME** uses `LimeTabularExplainer` on Random Forest to explain each individual URL prediction locally.

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/kashishhmehra/phishing-url-detection.git
cd phishing-url-detection
```

**2. Create a virtual environment**
```bash
python -m venv phishing_env
phishing_env\Scripts\activate   # Windows
source phishing_env/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
shap
lime
matplotlib
```

---

## Project Structure

```
phishing-url-detection/
├── app.py                        # Streamlit web app
├── phishing_xai_model.ipynb      # Training notebook
├── random_forest_model.pkl       # Trained RF model
├── logistic_regression_model.pkl # Trained LR model
├── scaler.pkl                    # StandardScaler for LR
├── feature_names.pkl             # Feature names list
├── requirements.txt              # Dependencies
└── README.md
```

---

## Research Paper

This project was built to support a research paper on **Explainable AI for Phishing Detection**, demonstrating that XAI methods like SHAP and LIME can make black-box ML models transparent and trustworthy for end users in cybersecurity applications.

---

## Author

**Kashish Mehra**
