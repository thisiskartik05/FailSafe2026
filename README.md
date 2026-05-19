# 🚨 FAILSAFE

## Early Student Failure Detection System

FAILSAFE is an explainable machine learning system designed to identify academically at-risk students before end-semester failure occurs.

The project combines:
- predictive machine learning,
- explainable AI,
- intervention recommendation systems,
- and interactive educational analytics dashboards

to help institutions move from **reactive grading** toward **proactive student support**.

---

# 🌍 Problem Statement

In most educational institutions, student failure is identified only after final examination results are released. By then, opportunities for meaningful intervention are minimal.

FAILSAFE addresses this problem by predicting student failure risk early using:
- behavioral indicators,
- attendance patterns,
- study habits,
- family background,
- social behavior,
- and academic history.

The goal is not merely prediction accuracy, but actionable intervention.

---

# 🧠 Features

## 📊 Early Risk Prediction

Uses an **XGBoost classifier** trained on:
- attendance,
- study time,
- alcohol consumption,
- prior failures,
- social activity,
- family relationships,
- and other student indicators.

Predicts whether a student is academically at risk.

---

## 🔍 Explainable AI with SHAP

Every prediction includes:
- top contributing features,
- feature direction,
- feature impact magnitude.

Example:
> “High absences increased failure risk by +0.18”

This makes predictions interpretable for faculty.

---

## 🩺 Rule-Based Intervention Engine

FAILSAFE converts predictions into actionable recommendations.

Examples:
- High absences → Attendance counselling
- Low study time → Structured study plan
- Multiple failures → Peer tutoring
- Alcohol indicators → Wellness referral

---

## 🖥 Interactive Streamlit Dashboard

### 1. Bulk Upload Dashboard
Faculty upload CSV files and receive:
- ranked risk predictions,
- probability scores,
- class labels.

### 2. Individual Risk Dashboard
Provides:
- SHAP explanation plots,
- student-specific analysis,
- personalized intervention recommendations.

---

# 🧮 Mathematical Foundations

## XGBoost Ensemble Learning

The prediction model is represented as:

\[
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
\]

where:
- \(f_k\) are individual decision trees,
- \(K\) is the number of boosting rounds,
- \(x_i\) is the feature vector of student \(i\).

---

## Gradient Boosting Objective

At iteration \(t\):

\[
L^{(t)} =
\sum_i
l\left(
y_i,
\hat{y}_i^{(t-1)} + f_t(x_i)
\right)
+
\Omega(f_t)
\]

where:
- \(l\) is the classification loss,
- \(\Omega\) is the regularization term.

---

## SHAP Explainability

SHAP decomposes predictions as:

\[
f(x)=\phi_0 + \sum_{i=1}^{M}\phi_i
\]

where:
- \(\phi_i\) represents feature contribution,
- \(M\) is the number of features.

This allows feature-level interpretability for every prediction.

---

# 📂 Dataset

## UCI Student Performance Dataset

- 395 student records
- 33 attributes
- demographic + behavioral + academic features

### Features Include:
- attendance
- study time
- internet access
- family background
- alcohol consumption
- free time
- past failures
- social activity

---

# 🎯 Target Variable

The target variable was constructed as:

\[
\text{at\_risk} =
\begin{cases}
1 & \text{if } G3 < 10 \\
0 & \text{otherwise}
\end{cases}
\]

To avoid target leakage:
- \(G1\),
- \(G2\),
- \(G3\)

were removed during training.

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.72 |
| AUC Score | 0.7097 |
| Weighted F1 Score | 0.70 |
| Precision (At Risk) | 0.62 |
| Recall (At Risk) | 0.38 |
| F1 Score (At Risk) | 0.48 |

---

# 🛠 Tech Stack

| Component | Technology |
|---|---|
| Machine Learning | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |
| Version Control | GitHub |

---

# 🏗 System Architecture

```text
Faculty CSV Upload
        ↓
Preprocessing & Encoding
        ↓
XGBoost Prediction
        ↓
SHAP Explainability
        ↓
Intervention Recommendation Engine
        ↓
Risk Dashboard
