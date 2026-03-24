# Diabetes Risk Prediction Pipeline

An advanced machine learning pipeline for early prediction of diabetes using clinical and lifestyle factors.

## 📁 Project Structure

```
hackatly/
├── diabetes_pipeline.py          # Main ML pipeline & model training
├── dataset_cleaned.csv           # Cleaned dataset (processed & validated)
├── dataset.csv                   # Original dataset (reference)
├── outputs/                      # Generated visualizations
│   ├── 01_pipeline_dashboard.png
│   ├── 02_feature_analysis.png
│   ├── 03_explainability.png
│   ├── 04_correlation_heatmap.png
│   └── 05_generalization.png
└── README.md                     # This file
```

## 🔧 Dataset Information

### `dataset_cleaned.csv` (Recommended)
- **Status**: Cleaned and clinically validated
- **Rows**: 768 records
- **Columns**: 13 features + 1 target
- **Data Quality**:
  - ✅ Removed synthetic determinism from SleepHours
  - ✅ Fixed Insulin missing values (30.5 placeholder imputation)
  - ✅ Validated physiological ranges
  - ✅ Clinically realistic correlations

### `dataset.csv` (Original)
- Raw dataset with synthetic artifacts
- Contains deterministic SleepHours-Outcome relationship
- Kept for reference only

## 📊 Features

### Clinical Features (7)
- **Glucose**: Fasting blood glucose (mg/dL)
- **BloodPressure**: Systolic blood pressure (mmHg)
- **BMI**: Body Mass Index (kg/m²)
- **Insulin**: Fasting insulin level (µIU/mL)
- **DiabetesPedigreeFunction**: Genetic risk factor
- **Age**: Patient age (years)
- **Pregnancies**: Number of pregnancies (for reference)

### Lifestyle Features (4)
- **SleepHours**: Average daily sleep (hours) - 4-10 range, stochastically related to outcome
- **ActivityLevel**: Physical activity level (0-2: low, moderate, high)
- **StressLevel**: Reported stress level (1-9 scale)
- **SugarIntake**: Daily sugar consumption (0-2: low, moderate, high)

### Target
- **Outcome**: Diabetes status (0 = non-diabetic, 1 = diabetic)

## 🚀 Quick Start

### Run the Pipeline
```bash
python diabetes_pipeline.py
```

This will:
1. Load and preprocess the cleaned dataset
2. Engineer clinical features
3. Train 4 models with hyperparameter tuning
4. Evaluate on both clinical-only and full-feature tracks
5. Generate 5 diagnostic visualizations
6. Display results and save figures

### Output Files
All visualizations are saved in the `outputs/` directory and displayed in real-time.

## 📈 Model Evaluation

The pipeline trains and compares:
- **Logistic Regression** (clinical baseline)
- **Random Forest** (with tuned hyperparameters)
- **Gradient Boosting** (with tuned hyperparameters)
- **HistGradientBoosting** (advanced gradient boosting)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (primary metric)
- 10-fold cross-validation AUC
- Feature importance & permutation importance
- Overfitting analysis

## 🔍 Key Findings

### Data Quality Improvements
- **Before**: Perfect separation in SleepHours (3-5h for diabetics, 6-10h for non-diabetics)
- **After**: Realistic stochastic relationship with age, stress, and activity adjustments

### Clinical Validity
- Early prediction focuses on modifiable risk factors
- Lifestyle features show realistic (non-deterministic) correlations
- Model designed for early intervention and risk stratification

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib

## 📝 Notes

- This dataset is designed for **early diabetes prediction**
- The cleaned version removes artificial data artifacts
- Lifestyle factors (sleep, activity, stress) are now realistically correlated with outcome
- Model suitable for clinical deployment with proper validation

## 🔐 Data Privacy

All data in this project is synthetic/anonymized and suitable for educational and research purposes.

---

**Last Updated**: 2026-03-24  
**Dataset Version**: Cleaned v1.0  
**Pipeline Version**: Final v3
