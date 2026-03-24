"""
================================================================================
  ADVANCED DIABETES RISK PREDICTION SYSTEM
  =========================================
  A production-ready ML system for early diabetes risk screening

  Features:
  - Multi-model comparison with hyperparameter tuning
  - SHAP-based explainability
  - Doctor-like clinical reports
  - Probabilistic future risk estimation
  - Strict train/test separation (no data leakage)

  Author: Clinical Data Science Team
  Version: 2.0.0
================================================================================
"""

# ============================================================================
#  IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier  # XGBoost alternative in sklearn
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# Try importing XGBoost, fallback to HistGradientBoosting
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. Using HistGradientBoostingClassifier as alternative.")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with 'pip install shap' for explainability.")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
#  CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration parameters."""
    DATA_PATH: str = "dataset_cleaned.csv"
    OUTPUT_DIR: str = "outputs"
    MODEL_DIR: str = "models"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20  # 80-20 split
    CV_FOLDS: int = 5  # Cross-validation folds for tuning

    # Clinical thresholds
    GLUCOSE_NORMAL: float = 100.0
    GLUCOSE_PREDIABETIC: float = 126.0
    BMI_NORMAL: float = 25.0
    BMI_OVERWEIGHT: float = 30.0


class RiskCategory(Enum):
    """Risk classification levels."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


# Color palette for visualizations
COLORS = {
    "bg": "#0F1117", "panel": "#1A1D27", "grid": "#2A2D3E",
    "text": "#E8E8F0", "subtext": "#9090B0",
    "blue": "#4FC3F7", "green": "#66BB6A",
    "orange": "#FFA726", "red": "#EF5350", "purple": "#7C5CBF"
}

# ============================================================================
#  1. DATA LOADING & PREPROCESSING
# ============================================================================

class DataProcessor:
    """Handles all data loading, preprocessing, and feature engineering."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.clinical_features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        self.lifestyle_features = [
            "SleepHours", "ActivityLevel", "StressLevel", "SugarIntake"
        ]

    def load_data(self, path: str = None) -> pd.DataFrame:
        """Load the diabetes dataset."""
        path = path or self.config.DATA_PATH
        df = pd.read_csv(path)

        print("=" * 70)
        print("  DATA LOADING")
        print("=" * 70)
        print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Diabetic cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
        print(f"  Non-diabetic: {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing: handle outliers and engineer features."""
        df = df.copy()

        print("\n  PREPROCESSING:")

        # 1. Handle outliers using winsorization (5th-95th percentile)
        outlier_cols = ["Insulin", "SkinThickness", "BMI",
                       "DiabetesPedigreeFunction", "BloodPressure"]
        for col in outlier_cols:
            if col in df.columns:
                lo, hi = df[col].quantile(0.05), df[col].quantile(0.95)
                df[col] = df[col].clip(lo, hi)
        print(f"    - Outliers clipped (5th-95th percentile)")

        # 2. Feature Engineering (clinically motivated)
        # Metabolic syndrome proxy
        df["Glucose_BMI"] = df["Glucose"] * df["BMI"] / 1000

        # Age-adjusted glycemia
        df["Age_Glucose"] = df["Age"] * df["Glucose"] / 1000

        # Insulin sensitivity (HOMA-IR proxy)
        df["Insulin_Sensitivity"] = df["Insulin"] / (df["Glucose"] + 1e-3)

        # Composite lifestyle risk score
        df["Lifestyle_Risk"] = (
            (2 - df["ActivityLevel"]) * 2 +  # Low activity = higher risk
            df["StressLevel"] +               # Higher stress = higher risk
            df["SugarIntake"] * 2 -           # Higher sugar = higher risk
            (df["SleepHours"] - 6).clip(-2, 2)  # Poor sleep = higher risk
        )

        print(f"    - Engineered features: Glucose_BMI, Age_Glucose, Insulin_Sensitivity, Lifestyle_Risk")

        return df

    def prepare_features(self, df: pd.DataFrame, use_lifestyle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target vector."""

        # Define feature set
        engineered = ["Glucose_BMI", "Age_Glucose", "Insulin_Sensitivity"]

        if use_lifestyle:
            self.feature_names = self.clinical_features + engineered + self.lifestyle_features + ["Lifestyle_Risk"]
        else:
            self.feature_names = self.clinical_features + engineered

        X = df[self.feature_names].values
        y = df["Outcome"].values

        print(f"\n  FEATURES SELECTED: {len(self.feature_names)}")
        print(f"    Clinical: {len(self.clinical_features) + len(engineered)}")
        if use_lifestyle:
            print(f"    Lifestyle: {len(self.lifestyle_features) + 1}")

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into training (80%) and testing (20%) sets.
        IMPORTANT: Stratified split to maintain class balance.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y  # Maintain class proportions
        )

        print(f"\n  DATA SPLIT (80/20, stratified):")
        print(f"    Training set: {len(y_train)} samples ({y_train.sum()} diabetic)")
        print(f"    Test set: {len(y_test)} samples ({y_test.sum()} diabetic)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        IMPORTANT: Fit ONLY on training data to prevent data leakage.
        """
        # Fit scaler on training data only
        self.scaler.fit(X_train)

        # Transform both sets
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"    Scaling: StandardScaler (fit on train only)")

        return X_train_scaled, X_test_scaled


# ============================================================================
#  2. MODEL TRAINING & HYPERPARAMETER TUNING
# ============================================================================

class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation."""

    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None

    def get_models_and_params(self) -> Dict[str, Tuple[Any, Dict]]:
        """Define models and their hyperparameter search spaces."""

        models_params = {
            "Logistic Regression": (
                LogisticRegression(
                    class_weight="balanced",
                    random_state=self.config.RANDOM_STATE,
                    max_iter=2000
                ),
                {
                    "C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                    "solver": ["lbfgs", "liblinear"]
                }
            ),

            "Random Forest": (
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=self.config.RANDOM_STATE,
                    n_jobs=-1
                ),
                {
                    "n_estimators": [100, 200, 300, 400],
                    "max_depth": [5, 7, 10, 12, None],
                    "min_samples_split": [5, 10, 15, 20],
                    "min_samples_leaf": [3, 5, 8, 10],
                    "max_features": ["sqrt", "log2"]
                }
            ),

            "Gradient Boosting": (
                GradientBoostingClassifier(
                    random_state=self.config.RANDOM_STATE
                ),
                {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "min_samples_split": [5, 10, 15]
                }
            ),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_params["XGBoost"] = (
                XGBClassifier(
                    random_state=self.config.RANDOM_STATE,
                    eval_metric="logloss",
                    use_label_encoder=False
                ),
                {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15],
                    "subsample": [0.7, 0.8, 0.9],
                    "colsample_bytree": [0.7, 0.8, 0.9]
                }
            )
        else:
            models_params["XGBoost (HistGB)"] = (
                HistGradientBoostingClassifier(
                    random_state=self.config.RANDOM_STATE,
                    early_stopping=True,
                    validation_fraction=0.15
                ),
                {
                    "max_iter": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15],
                    "min_samples_leaf": [10, 15, 20],
                    "l2_regularization": [0.0, 0.5, 1.0]
                }
            )

        return models_params

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        IMPORTANT: Tuning is done ONLY on training data using cross-validation.
        """
        print("\n" + "=" * 70)
        print("  HYPERPARAMETER TUNING (on training data only)")
        print("=" * 70)

        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )

        models_params = self.get_models_and_params()

        for name, (model, param_grid) in models_params.items():
            print(f"\n  Tuning: {name}...")

            # RandomizedSearchCV for efficient hyperparameter search
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,  # Number of parameter settings sampled
                cv=cv,
                scoring="roc_auc",  # Optimize for ROC-AUC
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )

            search.fit(X_train, y_train)

            self.best_params[name] = search.best_params_
            self.models[name] = search.best_estimator_

            print(f"    Best params: {search.best_params_}")
            print(f"    CV ROC-AUC: {search.best_score_:.4f}")

    def evaluate_models(
        self,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate all models on the TEST SET.
        IMPORTANT: Test set is used ONLY for final evaluation.
        """
        print("\n" + "=" * 70)
        print("  MODEL EVALUATION (on test set)")
        print("=" * 70)

        print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
        print("  " + "-" * 75)

        for name, model in self.models.items():
            # Predictions on test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            # ROC curve data
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Store results
            self.results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "confusion_matrix": cm,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "model": model
            }

            print(f"  {name:<25} {accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {roc_auc:>10.4f}")

        print("  " + "-" * 75)

        # Select best model based on ROC-AUC
        self.best_model_name = max(self.results, key=lambda n: self.results[n]["roc_auc"])
        self.best_model = self.results[self.best_model_name]["model"]

        best_auc = self.results[self.best_model_name]["roc_auc"]
        best_acc = self.results[self.best_model_name]["accuracy"]

        print(f"\n  BEST MODEL: {self.best_model_name}")
        print(f"    ROC-AUC: {best_auc:.4f}")
        print(f"    Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

        return self.results

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from the best model."""

        model = self.best_model

        # Get importance based on model type
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    def save_model(self, scaler: StandardScaler, feature_names: List[str]):
        """Save the best model and scaler."""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)

        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "scaler": scaler,
            "feature_names": feature_names,
            "results": self.results[self.best_model_name]
        }

        path = os.path.join(self.config.MODEL_DIR, "diabetes_model.joblib")
        joblib.dump(model_data, path)
        print(f"\n  Model saved: {path}")


# ============================================================================
#  3. EXPLAINABILITY MODULE (SHAP)
# ============================================================================

class Explainer:
    """SHAP-based model explainability."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def initialize(self, X_background: np.ndarray):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            print("  SHAP not available for explainability.")
            return

        # Sample background data for efficiency
        if len(X_background) > 100:
            idx = np.random.choice(len(X_background), 100, replace=False)
            X_background = X_background[idx]

        # Use TreeExplainer for tree-based models
        model_type = type(self.model).__name__
        if any(t in model_type for t in ["Forest", "Gradient", "XGB", "Hist"]):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background
            )

        print(f"  SHAP Explainer initialized: {type(self.explainer).__name__}")

    def explain(self, X_instance: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"available": False}

        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        try:
            shap_values = self.explainer.shap_values(X_instance)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_vals = np.array(shap_values[1]).flatten()
            elif hasattr(shap_values, 'values'):
                shap_vals = np.array(shap_values.values).flatten()
            else:
                shap_vals = np.array(shap_values).flatten()

            # Create contributions dictionary
            contributions = {}
            for feat, val in zip(self.feature_names, shap_vals):
                contributions[feat] = {
                    "value": float(val),
                    "direction": "increases risk" if val > 0 else "decreases risk",
                    "magnitude": abs(float(val))
                }

            # Sort by magnitude
            sorted_features = sorted(
                contributions.items(),
                key=lambda x: x[1]["magnitude"],
                reverse=True
            )

            return {
                "available": True,
                "contributions": contributions,
                "top_factors": sorted_features[:5]
            }

        except Exception as e:
            return {"available": False, "error": str(e)}


# ============================================================================
#  4. RISK ASSESSMENT & FUTURE PROJECTION
# ============================================================================

class RiskAssessor:
    """Clinical risk assessment and future risk projection."""

    def __init__(self, config: Config):
        self.config = config

    def categorize_risk(self, probability: float) -> Tuple[RiskCategory, str]:
        """Categorize risk based on probability."""
        if probability < 0.25:
            return RiskCategory.LOW, "Low risk - maintain healthy lifestyle"
        elif probability < 0.50:
            return RiskCategory.MODERATE, "Moderate risk - lifestyle modifications recommended"
        elif probability < 0.75:
            return RiskCategory.HIGH, "High risk - medical evaluation strongly advised"
        else:
            return RiskCategory.VERY_HIGH, "Very high risk - immediate medical consultation recommended"

    def estimate_future_risk(
        self,
        probability: float,
        age: float,
        glucose: float,
        bmi: float
    ) -> Dict[str, Any]:
        """
        Estimate PROBABILISTIC risk progression over time.

        IMPORTANT: This is NOT a diagnosis timeline. It's a probabilistic
        estimate based on epidemiological patterns.
        """
        risk_factors = []
        base_progression = 0.02  # 2% annual baseline increase

        # Adjust progression based on clinical indicators
        if glucose >= self.config.GLUCOSE_PREDIABETIC:
            base_progression += 0.03
            risk_factors.append("elevated glucose (diabetic range)")
        elif glucose >= self.config.GLUCOSE_NORMAL:
            base_progression += 0.015
            risk_factors.append("pre-diabetic glucose levels")

        if bmi >= self.config.BMI_OVERWEIGHT:
            base_progression += 0.02
            risk_factors.append("obesity (BMI >= 30)")
        elif bmi >= self.config.BMI_NORMAL:
            base_progression += 0.01
            risk_factors.append("overweight (BMI 25-30)")

        if age > 45:
            base_progression += 0.015
            risk_factors.append("age > 45 years")

        # Project future probabilities
        projections = {}
        for years in [1, 3, 5, 10]:
            # Logistic-like growth capped at 95%
            future_prob = min(0.95, probability + (base_progression * years * (1 - probability)))
            category, _ = self.categorize_risk(future_prob)
            projections[f"{years}_year"] = {
                "probability": round(future_prob, 3),
                "percentage": round(future_prob * 100, 1),
                "category": category.value
            }

        # Generate time horizon message
        if probability >= 0.70:
            horizon = "Risk is already elevated. Without intervention, diabetes development is likely within 1-3 years."
        elif probability >= 0.50:
            horizon = "At current trajectory, significant risk increase expected within 3-5 years."
        elif probability >= 0.30:
            horizon = "Moderate baseline risk. Without lifestyle changes, may progress to high risk within 5-7 years."
        else:
            horizon = "Currently low risk. Maintaining healthy lifestyle can preserve this status long-term."

        return {
            "current_probability": probability,
            "progression_rate_per_year": base_progression,
            "contributing_factors": risk_factors,
            "projections": projections,
            "time_horizon": horizon,
            "disclaimer": (
                "This is a probabilistic estimate based on population-level data. "
                "Individual outcomes vary significantly. This is NOT a diagnosis. "
                "Please consult a healthcare provider for personalized assessment."
            )
        }

    def generate_recommendations(self, probability: float, features: Dict[str, float]) -> List[str]:
        """Generate personalized preventive recommendations."""
        recommendations = []

        # General recommendation for elevated risk
        if probability >= 0.25:
            recommendations.append("Schedule regular HbA1c testing (every 6-12 months)")

        # Glucose-specific
        glucose = features.get("Glucose", 100)
        if glucose >= self.config.GLUCOSE_NORMAL:
            recommendations.append(
                f"Fasting glucose ({glucose:.0f} mg/dL) is elevated. Target: <100 mg/dL. "
                "Reduce refined carbohydrates and added sugars."
            )

        # BMI-specific
        bmi = features.get("BMI", 25)
        if bmi >= self.config.BMI_NORMAL:
            recommendations.append(
                f"BMI ({bmi:.1f}) indicates excess weight. "
                "Even 5-7% weight loss can significantly reduce diabetes risk."
            )

        # Activity level
        activity = features.get("ActivityLevel", 1)
        if activity < 2:
            recommendations.append(
                "Increase physical activity to at least 150 minutes/week of moderate exercise."
            )

        # Stress management
        stress = features.get("StressLevel", 5)
        if stress > 6:
            recommendations.append(
                "High stress can affect blood sugar. Consider stress management techniques."
            )

        # Diet
        sugar = features.get("SugarIntake", 1)
        if sugar >= 2:
            recommendations.append(
                "Reduce sugar intake. Limit sugary beverages and processed foods."
            )

        # Sleep
        sleep = features.get("SleepHours", 7)
        if sleep < 6:
            recommendations.append(
                "Insufficient sleep is linked to insulin resistance. Aim for 7-8 hours nightly."
            )

        # Family history
        pedigree = features.get("DiabetesPedigreeFunction", 0.3)
        if pedigree > 0.5:
            recommendations.append(
                "Family history indicates genetic predisposition. Regular monitoring is essential."
            )

        return recommendations


# ============================================================================
#  5. CLINICAL REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generates doctor-like clinical reports."""

    def __init__(self, risk_assessor: RiskAssessor, explainer: Explainer = None):
        self.risk_assessor = risk_assessor
        self.explainer = explainer

    def generate_report(
        self,
        probability: float,
        features: Dict[str, float],
        explanation: Dict = None
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical report."""

        # Risk categorization
        risk_category, risk_message = self.risk_assessor.categorize_risk(probability)

        # Future risk projection
        future_risk = self.risk_assessor.estimate_future_risk(
            probability,
            features.get("Age", 30),
            features.get("Glucose", 100),
            features.get("BMI", 25)
        )

        # Recommendations
        recommendations = self.risk_assessor.generate_recommendations(probability, features)

        # Clinical summary
        summary = self._generate_clinical_summary(probability, risk_category, features)

        # Feature analysis
        feature_analysis = self._analyze_key_features(features)

        return {
            "risk_probability": round(probability, 4),
            "risk_percentage": round(probability * 100, 1),
            "risk_category": risk_category.value,
            "risk_message": risk_message,
            "clinical_summary": summary,
            "key_indicators": feature_analysis,
            "future_risk": future_risk,
            "recommendations": recommendations,
            "warning_level": self._get_warning_level(risk_category),
            "explanation": explanation
        }

    def _generate_clinical_summary(
        self,
        probability: float,
        risk_category: RiskCategory,
        features: Dict[str, float]
    ) -> str:
        """Generate doctor-like clinical summary."""

        glucose = features.get("Glucose", 100)
        bmi = features.get("BMI", 25)
        age = features.get("Age", 30)

        # Opening statement based on risk level
        openings = {
            RiskCategory.LOW: (
                "Based on the clinical assessment, your metabolic profile suggests "
                "a low risk of developing type 2 diabetes."
            ),
            RiskCategory.MODERATE: (
                "Your current metabolic indicators suggest a moderate risk of "
                "developing type 2 diabetes."
            ),
            RiskCategory.HIGH: (
                "Clinical evaluation indicates a high risk of developing type 2 diabetes."
            ),
            RiskCategory.VERY_HIGH: (
                "Your metabolic profile indicates a very high risk of type 2 diabetes "
                "that requires immediate attention."
            )
        }
        opening = openings[risk_category]

        # Key contributors
        contributors = []
        if glucose >= 100:
            contributors.append("elevated fasting glucose")
        if bmi >= 25:
            contributors.append("elevated BMI")
        if age > 45:
            contributors.append("age-related metabolic factors")

        if contributors:
            contrib_text = f" Primary risk factors include {', '.join(contributors)}."
        else:
            contrib_text = " No major clinical risk factors identified."

        # Future outlook
        if probability >= 0.50:
            outlook = (
                " If current metabolic patterns persist, risk may increase "
                "significantly over the next 3-5 years."
            )
        elif probability >= 0.25:
            outlook = (
                " With appropriate lifestyle modifications, this risk can be "
                "substantially reduced."
            )
        else:
            outlook = (
                " Maintaining current healthy patterns will help preserve "
                "this favorable profile."
            )

        # Closing recommendation
        if risk_category in [RiskCategory.HIGH, RiskCategory.VERY_HIGH]:
            closing = " Consultation with a healthcare provider is strongly recommended."
        elif risk_category == RiskCategory.MODERATE:
            closing = " Lifestyle modifications and periodic monitoring are advisable."
        else:
            closing = " Continue current healthy lifestyle practices."

        return f"{opening}{contrib_text}{outlook}{closing}"

    def _analyze_key_features(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze key clinical indicators."""
        analysis = {}

        glucose = features.get("Glucose", 100)
        analysis["Glucose"] = {
            "value": glucose,
            "unit": "mg/dL",
            "status": "Normal" if glucose < 100 else "Pre-diabetic" if glucose < 126 else "Diabetic range",
            "concern": glucose >= 100
        }

        bmi = features.get("BMI", 25)
        analysis["BMI"] = {
            "value": round(bmi, 1),
            "unit": "kg/m²",
            "status": "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese",
            "concern": bmi >= 25
        }

        bp = features.get("BloodPressure", 80)
        analysis["Blood Pressure"] = {
            "value": bp,
            "unit": "mmHg (diastolic)",
            "status": "Normal" if bp < 80 else "Elevated" if bp < 90 else "High",
            "concern": bp >= 80
        }

        return analysis

    def _get_warning_level(self, risk_category: RiskCategory) -> Dict[str, str]:
        """Get warning level indicator."""
        levels = {
            RiskCategory.LOW: {"level": "GREEN", "action": "Continue monitoring"},
            RiskCategory.MODERATE: {"level": "YELLOW", "action": "Lifestyle changes needed"},
            RiskCategory.HIGH: {"level": "ORANGE", "action": "Medical evaluation advised"},
            RiskCategory.VERY_HIGH: {"level": "RED", "action": "Urgent medical attention"}
        }
        return levels[risk_category]

    def print_report(self, report: Dict[str, Any]):
        """Print formatted clinical report."""
        print("\n" + "=" * 70)
        print("  DIABETES RISK ASSESSMENT REPORT")
        print("=" * 70)

        # Risk summary
        print(f"\n  RISK ASSESSMENT:")
        print(f"    Probability: {report['risk_percentage']:.1f}%")
        print(f"    Category: {report['risk_category']}")
        print(f"    Warning Level: {report['warning_level']['level']}")
        print(f"    Action: {report['warning_level']['action']}")

        # Clinical summary
        print(f"\n  CLINICAL SUMMARY:")
        print("  " + "-" * 66)
        summary = report["clinical_summary"]
        # Word wrap at 66 characters
        words = summary.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= 66:
                line += word + " "
            else:
                print(f"  {line.strip()}")
                line = word + " "
        if line:
            print(f"  {line.strip()}")

        # Key indicators
        print(f"\n  KEY INDICATORS:")
        print("  " + "-" * 66)
        for name, data in report["key_indicators"].items():
            flag = " [CONCERN]" if data["concern"] else ""
            print(f"    {name}: {data['value']} {data['unit']} - {data['status']}{flag}")

        # Future risk
        print(f"\n  FUTURE RISK PROJECTION:")
        print("  " + "-" * 66)
        future = report["future_risk"]
        for period, proj in future["projections"].items():
            print(f"    {period.replace('_', ' ')}: {proj['percentage']:.1f}% ({proj['category']})")
        print(f"\n  {future['time_horizon']}")

        # Recommendations
        print(f"\n  RECOMMENDATIONS:")
        print("  " + "-" * 66)
        for i, rec in enumerate(report["recommendations"][:6], 1):
            print(f"    {i}. {rec}")

        # Disclaimer
        print(f"\n  DISCLAIMER:")
        print("  " + "-" * 66)
        print("  This assessment is for screening purposes only and does not")
        print("  constitute a medical diagnosis. Please consult a healthcare")
        print("  provider for clinical evaluation and personalized advice.")

        print("\n" + "=" * 70)


# ============================================================================
#  6. VISUALIZATION MODULE
# ============================================================================

class Visualizer:
    """Generate visualizations for model evaluation."""

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Set plot style
        plt.rcParams.update({
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["subtext"],
            "ytick.color": COLORS["subtext"],
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "legend.facecolor": COLORS["panel"],
            "legend.edgecolor": COLORS["grid"],
        })

    def plot_roc_curves(self, results: Dict[str, Dict], save: bool = True):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS["bg"])

        colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]

        # Random baseline
        ax.plot([0, 1], [0, 1], ":", color=COLORS["subtext"], lw=1.5,
                label="Random Classifier (AUC = 0.50)")

        # Plot each model
        for (name, r), color in zip(results.items(), colors):
            ax.plot(r["fpr"], r["tpr"], color=color, lw=2.5,
                   label=f"{name} (AUC = {r['roc_auc']:.3f})")

        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title(
            "ROC Curves - Model Comparison\n"
            "AUC = Probability that model ranks a diabetic patient higher than non-diabetic",
            fontweight="bold", fontsize=12
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        plt.tight_layout()

        if save:
            path = os.path.join(self.config.OUTPUT_DIR, "roc_curves.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
            print(f"  Saved: {path}")

        plt.close()

    def plot_confusion_matrices(self, results: Dict[str, Dict], y_test: np.ndarray, save: bool = True):
        """Plot confusion matrices for all models."""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5), facecolor=COLORS["bg"])

        if n_models == 1:
            axes = [axes]

        colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]

        for ax, (name, r), color in zip(axes, results.items(), colors):
            cm = r["confusion_matrix"]

            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list("custom", [COLORS["panel"], color])

            im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

            # Add text annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else COLORS["text"],
                           fontsize=14, fontweight="bold")

            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title(f"{name}\nAcc={r['accuracy']:.3f} | AUC={r['roc_auc']:.3f}",
                        fontsize=10, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=8)
            ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=8)

        plt.tight_layout()

        if save:
            path = os.path.join(self.config.OUTPUT_DIR, "confusion_matrices.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
            print(f"  Saved: {path}")

        plt.close()

    def plot_feature_importance(self, importance_df: pd.DataFrame, save: bool = True):
        """Plot feature importance chart."""
        if importance_df.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS["bg"])

        top_n = min(15, len(importance_df))
        df = importance_df.head(top_n)

        # Color by feature type
        lifestyle_features = ["SleepHours", "ActivityLevel", "StressLevel",
                            "SugarIntake", "Lifestyle_Risk"]
        colors = [COLORS["orange"] if f in lifestyle_features else COLORS["blue"]
                 for f in df["feature"]]

        y_pos = range(top_n)
        ax.barh(y_pos, df["importance"].values[::-1], color=colors[::-1], alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["feature"].values[::-1], fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Feature Importance (Best Model)", fontweight="bold", fontsize=12)
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["blue"], label="Clinical"),
            Patch(facecolor=COLORS["orange"], label="Lifestyle")
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()

        if save:
            path = os.path.join(self.config.OUTPUT_DIR, "feature_importance.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
            print(f"  Saved: {path}")

        plt.close()

    def plot_model_comparison(self, results: Dict[str, Dict], save: bool = True):
        """Plot metric comparison across models."""
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["bg"])

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]

        x = np.arange(len(metrics))
        width = 0.2

        for i, (name, r) in enumerate(results.items()):
            values = [r[m] for m in metrics]
            ax.bar(x + i * width - 0.3, values, width, color=colors[i],
                  label=name, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Model Performance Comparison (Test Set)", fontweight="bold", fontsize=12)
        ax.axhline(0.85, color=COLORS["red"], ls="--", lw=1.5, alpha=0.7, label="Target (0.85)")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.config.OUTPUT_DIR, "model_comparison.png")
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
            print(f"  Saved: {path}")

        plt.close()


# ============================================================================
#  7. MAIN PREDICTION SYSTEM
# ============================================================================

class DiabetesPredictor:
    """Main diabetes risk prediction system."""

    def __init__(self):
        self.config = Config()
        self.processor = DataProcessor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.risk_assessor = RiskAssessor(self.config)
        self.explainer: Optional[Explainer] = None
        self.report_generator: Optional[ReportGenerator] = None
        self.visualizer = Visualizer(self.config)
        self.is_trained = False
        self.X_train = None
        self.y_test = None

    def train(self, data_path: str = None, use_lifestyle: bool = False):
        """
        Train the diabetes prediction system.

        Args:
            data_path: Path to dataset (default: dataset_cleaned.csv)
            use_lifestyle: Include lifestyle features (default: False for clinical-only)
        """
        print("\n" + "=" * 70)
        print("  DIABETES RISK PREDICTION SYSTEM - TRAINING PIPELINE")
        print("=" * 70)

        # 1. Load data
        df = self.processor.load_data(data_path)

        # 2. Preprocess
        df = self.processor.preprocess(df)

        # 3. Prepare features
        X, y = self.processor.prepare_features(df, use_lifestyle=use_lifestyle)

        # 4. Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = self.processor.split_data(X, y)
        self.y_test = y_test

        # 5. Scale features (fit on train only)
        X_train_scaled, X_test_scaled = self.processor.scale_features(X_train, X_test)
        self.X_train = X_train_scaled

        # 6. Hyperparameter tuning (on training data only)
        self.trainer.tune_hyperparameters(X_train_scaled, y_train)

        # 7. Evaluate models (on test data only)
        results = self.trainer.evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # 8. Get feature importance
        importance_df = self.trainer.get_feature_importance(self.processor.feature_names)

        if not importance_df.empty:
            print("\n  TOP FEATURE IMPORTANCES:")
            print(importance_df.head(10).to_string(index=False))

        # 9. Initialize explainability
        if SHAP_AVAILABLE:
            self.explainer = Explainer(self.trainer.best_model, self.processor.feature_names)
            self.explainer.initialize(X_train_scaled)

        # 10. Initialize report generator
        self.report_generator = ReportGenerator(self.risk_assessor, self.explainer)

        # 11. Generate visualizations
        print("\n  GENERATING VISUALIZATIONS:")
        self.visualizer.plot_roc_curves(results)
        self.visualizer.plot_confusion_matrices(results, y_test)
        self.visualizer.plot_feature_importance(importance_df)
        self.visualizer.plot_model_comparison(results)

        # 12. Save model
        self.trainer.save_model(self.processor.scaler, self.processor.feature_names)

        self.is_trained = True

        # Print ROC-AUC explanation
        self._explain_roc_auc()

        return results, importance_df

    def predict(self, user_input: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for user input.

        Args:
            user_input: Dictionary with feature values

        Returns:
            Comprehensive risk assessment report
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Fill defaults
        defaults = {
            "Pregnancies": 0, "Glucose": 100, "BloodPressure": 70,
            "SkinThickness": 20, "Insulin": 80, "BMI": 25,
            "DiabetesPedigreeFunction": 0.3, "Age": 30,
            "SleepHours": 7, "ActivityLevel": 1, "StressLevel": 5, "SugarIntake": 1
        }
        for key, default in defaults.items():
            if key not in user_input:
                user_input[key] = default

        # Compute engineered features
        user_input["Glucose_BMI"] = user_input["Glucose"] * user_input["BMI"] / 1000
        user_input["Age_Glucose"] = user_input["Age"] * user_input["Glucose"] / 1000
        user_input["Insulin_Sensitivity"] = user_input["Insulin"] / (user_input["Glucose"] + 1e-3)
        user_input["Lifestyle_Risk"] = (
            (2 - user_input["ActivityLevel"]) * 2 +
            user_input["StressLevel"] +
            user_input["SugarIntake"] * 2 -
            (user_input["SleepHours"] - 6)
        )

        # Build feature vector
        feature_vector = np.array([
            user_input[f] for f in self.processor.feature_names
        ]).reshape(1, -1)

        # Scale
        feature_vector_scaled = self.processor.scaler.transform(feature_vector)

        # Predict probability
        probability = self.trainer.best_model.predict_proba(feature_vector_scaled)[0, 1]

        # Get SHAP explanation
        explanation = None
        if self.explainer:
            explanation = self.explainer.explain(feature_vector_scaled)

        # Generate report
        report = self.report_generator.generate_report(
            probability, user_input, explanation
        )

        return report

    def interactive_mode(self):
        """Run interactive prediction mode."""
        if not self.is_trained:
            print("Error: Model not trained. Call train() first.")
            return

        print("\n" + "=" * 70)
        print("  DIABETES RISK ASSESSMENT - Interactive Mode")
        print("=" * 70)
        print("\nEnter patient information (press Enter for defaults):\n")

        def get_input(prompt: str, default: float, dtype=float) -> float:
            while True:
                try:
                    value = input(f"  {prompt} [{default}]: ").strip()
                    return dtype(value) if value else default
                except ValueError:
                    print("    Invalid input. Please enter a number.")

        user_input = {}

        print("CLINICAL INDICATORS:")
        user_input["Glucose"] = get_input("Fasting Glucose (mg/dL)", 100)
        user_input["BMI"] = get_input("BMI (kg/m²)", 25)
        user_input["Age"] = get_input("Age (years)", 30, int)
        user_input["BloodPressure"] = get_input("Blood Pressure - diastolic (mmHg)", 70)
        user_input["Insulin"] = get_input("Insulin (mu U/ml)", 80)
        user_input["Pregnancies"] = get_input("Number of Pregnancies", 0, int)
        user_input["SkinThickness"] = get_input("Skin Thickness (mm)", 20)
        user_input["DiabetesPedigreeFunction"] = get_input("Diabetes Pedigree (0-2.5)", 0.3)

        print("\nLIFESTYLE FACTORS:")
        user_input["SleepHours"] = get_input("Average Sleep (hours/night)", 7)
        user_input["ActivityLevel"] = get_input("Activity Level (0=low, 1=med, 2=high)", 1, int)
        user_input["StressLevel"] = get_input("Stress Level (1-10)", 5)
        user_input["SugarIntake"] = get_input("Sugar Intake (0=low, 1=med, 2=high)", 1, int)

        print("\nAnalyzing risk profile...")
        report = self.predict(user_input)
        self.report_generator.print_report(report)

        return report

    def load_model(self, model_path: str):
        """
        Load a pre-trained model from file.

        Args:
            model_path: Path to the saved model file (.joblib)
        """
        print(f"\n  Loading model from: {model_path}")

        model_data = joblib.load(model_path)

        # Restore model components
        self.trainer.best_model = model_data["model"]
        self.trainer.best_model_name = model_data["model_name"]
        self.processor.scaler = model_data["scaler"]
        self.processor.feature_names = model_data["feature_names"]
        self.trainer.results = {model_data["model_name"]: model_data["results"]}

        # Initialize explainer with loaded model
        if SHAP_AVAILABLE:
            self.explainer = Explainer(self.trainer.best_model, self.processor.feature_names)
            try:
                self.explainer.explainer = shap.TreeExplainer(self.trainer.best_model)
            except Exception:
                pass  # SHAP optional

        # Initialize report generator
        self.report_generator = ReportGenerator(self.risk_assessor, self.explainer)

        self.is_trained = True
        print(f"  Model loaded: {model_data['model_name']}")
        print(f"  Features: {len(self.processor.feature_names)}")

    def _explain_roc_auc(self):
        """Print ROC-AUC explanation."""
        print("\n" + "=" * 70)
        print("  UNDERSTANDING ROC-AUC")
        print("=" * 70)
        print("""
  ROC-AUC (Receiver Operating Characteristic - Area Under Curve):

  WHAT IT MEASURES:
  - How well the model distinguishes between diabetic and non-diabetic
    patients across ALL possible classification thresholds.

  - It answers: "If I randomly pick one diabetic and one non-diabetic
    patient, what's the probability the model correctly ranks the
    diabetic patient as higher risk?"

  INTERPRETATION:
  - AUC = 0.50: Random guessing (no predictive power)
  - AUC = 0.70: Acceptable discrimination
  - AUC = 0.80: Good discrimination
  - AUC = 0.85: Very good discrimination (our target)
  - AUC = 0.90+: Excellent discrimination

  WHY IT'S IMPORTANT FOR MEDICAL SCREENING:
  - Threshold-independent: Evaluates model's ranking ability
  - Handles class imbalance: Remains reliable with skewed data
  - Clinically meaningful: Directly measures risk stratification
""")
        print("=" * 70)


# ============================================================================
#  8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Initialize system
    predictor = DiabetesPredictor()

    # Train model (clinical features only for reliability)
    results, importance_df = predictor.train(use_lifestyle=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    best_name = predictor.trainer.best_model_name
    best_result = results[best_name]

    print(f"\n  Best Model: {best_name}")
    print(f"  Test Set Performance:")
    print(f"    - Accuracy:  {best_result['accuracy']*100:.2f}%")
    print(f"    - Precision: {best_result['precision']*100:.2f}%")
    print(f"    - Recall:    {best_result['recall']*100:.2f}%")
    print(f"    - F1 Score:  {best_result['f1']*100:.2f}%")
    print(f"    - ROC-AUC:   {best_result['roc_auc']:.4f}")

    # Example prediction
    print("\n" + "=" * 70)
    print("  EXAMPLE PREDICTION")
    print("=" * 70)

    example_patient = {
        "Pregnancies": 3,
        "Glucose": 145,
        "BloodPressure": 82,
        "SkinThickness": 30,
        "Insulin": 90,
        "BMI": 32.5,
        "DiabetesPedigreeFunction": 0.55,
        "Age": 48,
        "SleepHours": 6,
        "ActivityLevel": 0,
        "StressLevel": 7,
        "SugarIntake": 2
    }

    print("\n  Patient Profile:")
    for key, value in example_patient.items():
        print(f"    {key}: {value}")

    report = predictor.predict(example_patient)
    predictor.report_generator.print_report(report)

    # Usage instructions
    print("\n" + "=" * 70)
    print("  USAGE")
    print("=" * 70)
    print("""
  # Train the model:
  predictor = DiabetesPredictor()
  predictor.train()

  # Make a prediction:
  report = predictor.predict({
      "Glucose": 140,
      "BMI": 30,
      "Age": 45,
      ...
  })

  # Interactive mode:
  predictor.interactive_mode()
""")
    print("=" * 70)

    return predictor


if __name__ == "__main__":
    predictor = main()
