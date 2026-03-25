# 📘 GlycoSense: Technical Documentation & Theoretical Framework

## 1. Project Overview & Architecture

### 1.1 Problem Statement
Diabetes Mellitus (Type 2) is a chronic metabolic condition often undiagnosed until complications arise. Traditional screening methods require invasive blood tests (HbA1c, OGTT) and clinical visits. GlycoSense aims to provide an accessible, non-invasive **pre-screening tool** using machine learning to estimate risk and key biomarkers based on self-reported and sensor-derived data.

### 1.2 System Architecture
The system follows a modular "User-to-Cloud-to-AI" pipeline:

1.  **Input Layer**:
    *   **Frontend**: React-inspired vanilla JS single-page application (SPA).
    *   **Sensors**: Optional integration with Arduino-based hardware (RGB sensor for urine glucose, accelerometers for sleep/activity).
2.  **Processing Layer (Flask Backend)**:
    *   **Data Mapping**: Converts user-friendly inputs (e.g., "High Sugar Intake") into numerical vectors.
    *   **Smart Imputation**: Fills missing clinical values using physiological heuristics (see Section 3.2).
3.  **Intelligence Layer (ML Core)**:
    *   **Risk Classifier**: Ensemble model predicting binary diabetes probability.
    *   **Biomarker Regressor**: Random Forest model estimating continuous HbA1c levels.
4.  **Presentation Layer**:
    *   Dynamic Gauge Charts (Risk Score).
    *   Probabilistic Future Projections (1, 3, 5, 10 years).

---

## 2. Machine Learning Methodology

### 2.1 Model Selection Rationale

We evaluated several algorithms based on the Pima Indians Diabetes Dataset (N=768) characteristics:
*   **Logistic Regression**: Used as a baseline. Good for interpretability (coefficients) but struggles with non-linear interactions (e.g., BMI impact varies by Age).
*   **Neural Networks (MLP)**: Discarded. Deep learning tends to overfit on small tabular datasets (<10k rows) and lacks inherent interpretability.
*   **Tree-Based Ensembles (Selected)**:
    *   **Random Forest**: Handles non-linear features well, robust to outliers, and reduces variance through bagging.
    *   **Gradient Boosting (GBM/XGBoost)**: Reduces bias by sequentially correcting errors of previous trees. Generally yields the highest accuracy on tabular data.

**Winner**: The system dynamically selects the best-performing ensemble (Random Forest or Gradient Boosting) based on cross-validation F1-scores during training.

### 2.2 Training Strategy
*   **Stratified K-Fold Cross-Validation (k=5)**: Ensures each fold has the same proportion of diabetic/non-diabetic cases as the full dataset. This prevents bias if the data is imbalanced (e.g., 35% diabetic).
*   **Hyperparameter Tuning**: We use `RandomizedSearchCV` to optimize:
    *   `n_estimators` (Number of trees)
    *   `max_depth` (Tree complexity)
    *   `min_samples_split` (Regularization)
    *   `class_weight` (Handling imbalance)

### 2.3 Performance Metrics
We prioritize **Recall (Sensitivity)** over Precision.
*   *Why?* In medical screening, a **False Negative** (missing a diabetic case) is dangerous. A **False Positive** (flagging a healthy person) just leads to a confirmatory blood test.
*   **Target Metrics**:
    *   Accuracy: ~78-85%
    *   ROC-AUC: >0.85 (Excellent discrimination)

---

## 3. Advanced Feature Engineering & Imputation

### 3.1 The "Missing Data" Challenge
Users rarely know their exact Insulin levels or Skin Thickness. Dropping these rows would waste data. Using the mean/median is inaccurate (a 120kg person has different insulin needs than a 60kg person).

### 3.2 Smart Imputation Logic
We derived formulas from clinical literature to estimate missing values:

*   **Insulin Estimation (HOMA-IR Proxy)**:
    $$ \text{Insulin} \approx f(\text{Glucose}, \text{BMI}) $$
    *   Logic: High BMI + High Glucose $\rightarrow$ Insulin Resistance $\rightarrow$ Higher fasting insulin.
    *   Implementation: Base 10 uU/mL + penalties for obesity (BMI>30) and hyperglycemia (>100 mg/dL).

*   **Skin Thickness (Subcutaneous Fat Proxy)**:
    $$ \text{SkinThickness} \approx 0.5 \times \text{BMI} + 0.1 \times \text{Waist} - 5 $$
    *   Logic: Skin fold thickness correlates strongly with overall body fat percentage.

*   **Pedigree Function (Genetic Risk)**:
    *   Standard dataset uses a specific formula. We approximate it by assigning weights to relatives:
        *   Parent: +0.5
        *   Grandparent: +0.25
        *   Sibling/Other: +0.1 - 0.2

---

## 4. The HbA1c Estimation Model (`hba1c_model.py`)

### 4.1 Theoretical Basis
HbA1c (Glycated Hemoglobin) reflects average blood sugar over 3 months. It is the gold standard for diagnosis.

### 4.2 Regression Approach
Instead of classification (Yes/No), we built a **Random Forest Regressor** to predict the *continuous* HbA1c value.

*   **Input Features**: All risk factors + engineered features (Glucose*BMI interaction).
*   **Target**: HbA1c percentage (e.g., 5.7%).

### 4.3 The "Medical Reality Check" (Post-Processing)
Pure ML models can hallucinate. A user with normal fasting glucose (85 mg/dL) might be predicted as having high HbA1c (6.0%) due to statistical noise in other features (e.g., Age).

To fix this, we implemented a **physiological constraint**:
$$ \text{Expected A1c} \approx \frac{\text{Fasting Glucose} + 77}{33} $$
*   *Correction*: If the model's prediction deviates significantly (>0.5%) from this physiological baseline—especially for healthy BMI individuals—we dampen the prediction towards the baseline.
*   *Result*: Drastically reduced False Positives for healthy users.

---

## 5. Future Risk Projection

### 5.1 Probabilistic Forecasting
We don't just predict "Current Risk." We estimate risk progression over time.

*   **Methodology**:
    *   We simulate "aging" the user profile (Age + N years).
    *   We apply a "metabolic drift" factor (slight increase in BMI/BP/Glucose over time based on population averages).
    *   We re-run the risk classifier on these projected profiles for T+1, T+3, T+5, and T+10 years.
*   **Value**: This demonstrates the *trajectory* of health, motivating early lifestyle intervention.

---

## 6. Hardware Integration (Sensors)

### 6.1 Glucose Sensor (Arduino)
*   **Principle**: Benedict's Reagent Reaction.
    *   Urine glucose + Reagent $\xrightarrow{\Delta}$ Color Change (Blue $\rightarrow$ Green/Orange/Red).
*   **Detection**: TCS3200/34725 RGB Sensor reads the color of the reacted solution.
*   **Calibration**: We map RGB values to glucose concentrations using a lookup table derived from standard solutions.

### 6.2 Data Fusion
*   Sensor data is logged to `sensor_data.csv`.
*   The system calculates a **Weighted Average**:
    $$ \text{Final Glucose} = 0.7 \times \text{Sensor Avg} + 0.3 \times \text{Self Report} $$
    *   *Why?* Measured data (Sensor) is prioritized over subjective recall (Self-report), but we keep the self-report as a stabilizer against sensor noise.

---

## 7. Conclusion

GlycoSense bridges the gap between basic online calculators and clinical diagnostics. By combining **Ensemble Learning**, **Clinical Heuristics**, and **Physiological Constraints**, it offers a medically grounded, explainable, and accessible screening tool.
