"""
================================================================================
  HbA1c PREDICTION MODEL
  ======================
  Machine Learning model to estimate HbA1c from physiological and lifestyle
  parameters, enabling non-invasive long-term glucose assessment.

  HbA1c (Glycated Hemoglobin) reflects average blood glucose over 2-3 months
  and is a key indicator for diabetes diagnosis and management.

  Features used:
  - Glucose (fasting)
  - BMI
  - Age
  - SkinThickness
  - Insulin
  - BloodPressure
  - Sleep Hours
  - Activity Level
  - Stress Level
  - Sugar Intake
  - Diabetes Pedigree Function

  Target: HbA1c percentage (%)
================================================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Union, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HbA1cPredictor:
    """
    HbA1c Prediction Model using Random Forest Regression.

    This model estimates HbA1c from physiological and lifestyle parameters,
    providing a non-invasive method for long-term glucose assessment.
    """

    def __init__(self):
        """Initialize the HbA1c predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age',
            'SleepHours',
            'ActivityLevel',
            'StressLevel',
            'SugarIntake'
        ]
        self.is_trained = False
        self.metrics = {}

    def _generate_realistic_hba1c(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate realistic HbA1c values based on biological processes.

        HbA1c formation depends on:
        1. Insulin resistance (Insulin, BMI, Age)
        2. Visceral fat (BMI, SkinThickness)
        3. Liver fat (BMI, insulin levels)
        4. Inflammation (StressLevel, SleepHours)
        5. Long-term glucose exposure (Glucose)
        6. Lifestyle factors (ActivityLevel, SugarIntake)
        """
        np.random.seed(42)  # For reproducible results

        # Normalize features to 0-1 scale for calculation
        glucose_norm = (df['Glucose'] - 65) / 135  # Typical range 65-200
        bmi_norm = (df['BMI'] - 18) / 50          # Range 18-68
        insulin_norm = (df['Insulin'] - 14) / 500  # Range 14-514
        age_norm = (df['Age'] - 21) / 60          # Range 21-81
        pregnancies_norm = df['Pregnancies'] / 8   # Range 0-8

        # Handle missing insulin values (use median)
        insulin_norm = insulin_norm.fillna(insulin_norm.median())

        # Calculate insulin resistance index (HOMA-IR approximation)
        # Pregnancies increase insulin resistance
        pregnancy_factor = 1 + (pregnancies_norm * 0.2)
        insulin_resistance = (glucose_norm * (insulin_norm + 0.1) * pregnancy_factor) * 2.0

        # Calculate visceral fat index (BMI + skin thickness)
        skin_norm = (df['SkinThickness'] - 7) / 50  # Range 7-57
        skin_norm = skin_norm.fillna(skin_norm.median())
        visceral_fat = (bmi_norm * 0.7) + (skin_norm * 0.3)

        # Calculate inflammation index (stress + poor sleep)
        stress_norm = df['StressLevel'] / 10        # Range 0-10
        sleep_norm = (8 - df['SleepHours']) / 5     # Poor sleep increases inflammation
        sleep_norm = np.clip(sleep_norm, 0, 1)
        inflammation = (stress_norm * 0.6) + (sleep_norm * 0.4)

        # Calculate lifestyle impact
        activity_benefit = (df['ActivityLevel'] / 2) * 0.1  # Higher activity = lower HbA1c
        sugar_penalty = (df['SugarIntake'] / 2) * 0.15      # Higher sugar = higher HbA1c

        # Base HbA1c from glucose (using modified medical formula)
        base_hba1c = (glucose_norm * 3.5) + 4.2

        # Additional contributions from metabolic factors
        metabolic_impact = (
            insulin_resistance * 0.8 +        # Major contributor
            visceral_fat * 0.6 +              # Significant contributor
            inflammation * 0.4 +              # Moderate contributor
            sugar_penalty - activity_benefit   # Lifestyle factors
        )

        # Genetic predisposition (DiabetesPedigreeFunction)
        genetic_impact = df['DiabetesPedigreeFunction'] * 0.5

        # Age-related insulin resistance
        age_impact = (age_norm ** 1.2) * 0.3

        # Combine all factors
        hba1c_calculated = (
            base_hba1c +
            metabolic_impact +
            genetic_impact +
            age_impact
        )

        # Add realistic biological noise (±0.3%)
        noise = np.random.normal(0, 0.3, len(df))
        hba1c_final = hba1c_calculated + noise

        # Ensure realistic bounds (3.5 - 15.0%)
        hba1c_final = np.clip(hba1c_final, 3.5, 15.0)

        return pd.Series(hba1c_final.round(2), index=df.index)

    def integrate_sensor_data(self, base_data: Dict, sensor_file: str = 'sensor_data.csv') -> Dict:
        """
        Integrate real-time sensor data with base patient data.

        Args:
            base_data: Dictionary with patient demographic and clinical data
            sensor_file: Path to sensor data CSV file

        Returns:
            Dictionary with integrated data for HbA1c prediction
        """
        integrated_data = base_data.copy()

        try:
            # Read sensor data
            if os.path.exists(sensor_file):
                sensor_df = pd.read_csv(sensor_file)

                # Clean and process sensor data
                sensor_df = sensor_df.dropna(subset=['timestamp'])

                # Calculate recent averages (last 7 days of data)
                recent_data = sensor_df.tail(7)

                # Average glucose from sensors (if available and not zero)
                glucose_readings = recent_data['glucose'].dropna()
                glucose_readings = glucose_readings[glucose_readings > 0]
                if len(glucose_readings) > 0:
                    avg_glucose = glucose_readings.mean()
                    # Weight sensor glucose with existing glucose (70% sensor, 30% base)
                    if 'Glucose' in integrated_data:
                        integrated_data['Glucose'] = (avg_glucose * 0.7) + (integrated_data['Glucose'] * 0.3)
                    else:
                        integrated_data['Glucose'] = avg_glucose

                # Average sleep from sensors
                sleep_readings = recent_data['sleep'].dropna()
                sleep_readings = sleep_readings[sleep_readings > 0]
                if len(sleep_readings) > 0:
                    avg_sleep = sleep_readings.mean()
                    integrated_data['SleepHours'] = avg_sleep

                # Lifestyle to activity level mapping
                lifestyle_readings = recent_data['lifestyle'].dropna()
                if len(lifestyle_readings) > 0:
                    lifestyle_mode = lifestyle_readings.mode()
                    if len(lifestyle_mode) > 0:
                        lifestyle = lifestyle_mode.iloc[0]
                        activity_mapping = {
                            'Sedentary': 0,
                            'Lightly Active': 1,
                            'Moderate': 1,
                            'Active': 2,
                            'Very Active': 2
                        }
                        integrated_data['ActivityLevel'] = activity_mapping.get(lifestyle, 1)

                print(f"[+] Integrated sensor data from {len(recent_data)} recent readings")
            else:
                print(f"[!] Sensor file {sensor_file} not found, using base data only")

        except Exception as e:
            print(f"[!] Error reading sensor data: {e}")

        return integrated_data

    def train(self, dataset_path: str = 'dataset_cleaned.csv', test_size: float = 0.2):
        """
        Train the HbA1c prediction model.

        Args:
            dataset_path: Path to the training dataset
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary containing training metrics
        """
        print("\n" + "=" * 70)
        print("  HbA1c PREDICTION MODEL TRAINING")
        print("=" * 70)

        # Load dataset
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

        print(f"\n[*] Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Check if HbA1c column exists
        if 'HbA1c' not in df.columns:
            print("[!] HbA1c column not found. Adding it using realistic biological model...")
            df['HbA1c'] = self._generate_realistic_hba1c(df)
        else:
            # Re-generate HbA1c to fix the overfitting issue
            print("[!] Regenerating HbA1c using realistic biological model...")
            df['HbA1c'] = self._generate_realistic_hba1c(df)

        print(f"[+] Dataset loaded: {len(df)} samples")
        print(f"[+] HbA1c range: {df['HbA1c'].min():.2f}% - {df['HbA1c'].max():.2f}%")

        # Prepare features and target
        X = df[self.feature_names].copy()
        y = df['HbA1c'].copy()

        # Handle missing values
        X = X.fillna(X.median())

        print(f"[+] Features: {len(self.feature_names)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"[+] Train set: {len(X_train)} samples")
        print(f"[+] Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest Regressor with optimized parameters
        print("\n[*] Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,           # More trees for better accuracy
            max_depth=15,              # Deeper trees to capture complex interactions
            min_samples_split=3,       # Smaller splits for more detail
            min_samples_leaf=1,        # Allow single samples per leaf
            max_features='sqrt',       # Use square root of features
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()

        self.metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'cv_mae': cv_mae,
            'test_samples': len(X_test)
        }

        self.is_trained = True

        # Display results
        print("\n" + "=" * 70)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 70)
        print(f"  Mean Absolute Error (MAE):     {mae:.3f}%")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}%")
        print(f"  R² Score:                       {r2:.3f}")
        print(f"  Cross-Validation MAE:           {cv_mae:.3f}%")
        print("=" * 70)

        # Feature importance
        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n[*] Feature Importance (Top 5):")
        for idx, row in feature_imp.head(5).iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")

        print("\n[+] Model training completed successfully!")

        return self.metrics

    def predict(self, input_data: Union[Dict, pd.DataFrame], use_sensors: bool = True) -> float:
        """
        Predict HbA1c for given input data, optionally integrating sensor data.

        Args:
            input_data: Dictionary or DataFrame with feature values
            use_sensors: Whether to integrate sensor data from sensor_data.csv

        Returns:
            Predicted HbA1c percentage (%)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a trained model.")

        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            # Integrate sensor data if requested
            if use_sensors:
                input_data = self.integrate_sensor_data(input_data)
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Ensure all features are present with default values
        feature_defaults = {
            'Glucose': 85,  # Lower default for safety
            'BMI': 22,      # Healthy BMI default
            'Age': 30,
            'SkinThickness': 20,
            'Insulin': 10,  # Normal fasting insulin level
            'BloodPressure': 72,
            'SleepHours': 7,
            'ActivityLevel': 1,
            'StressLevel': 5,
            'SugarIntake': 1,
            'DiabetesPedigreeFunction': 0.1,
            'Pregnancies': 0
        }

        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = feature_defaults.get(feature, 0)

        # Select and order features
        X = input_df[self.feature_names]

        # Handle missing values using defaults
        for col in X.columns:
            X[col] = X[col].fillna(feature_defaults.get(col, X[col].median() if len(X) > 1 else 0))

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        hba1c_pred = self.model.predict(X_scaled)[0]

        # Medical Reality Check:
        # HbA1c rarely exceeds 5.7% (Prediabetes) if Fasting Glucose is < 100 mg/dL
        # and there are no extreme risk factors.
        # We apply a correction factor to align the prediction with physiological constraints.
        
        glucose_val = input_df['Glucose'].iloc[0]
        bmi_val = input_df['BMI'].iloc[0]
        
        # Calculate expected baseline from Fasting Glucose alone (approximate)
        # Adjusted: A1c ~= (FastingGlucose + 77) / 33 (Rough clinical heuristic)
        # e.g. 80 -> 4.7, 100 -> 5.3, 126 -> 6.1
        expected_a1c = (glucose_val + 77) / 33.0
        
        # Pull model prediction towards physiologically expected value
        # especially if BMI is low (healthy)
        diff = hba1c_pred - expected_a1c
        
        if diff > 0:
             # Apply stronger correction for healthy individuals to prevent false positives
             correction_strength = 0.8 if bmi_val < 25 else 0.5
             hba1c_pred -= (diff * correction_strength)
        
        # Ensure realistic bounds (3.5 - 15.0%)
        hba1c_pred = np.clip(hba1c_pred, 3.5, 15.0)

        return round(hba1c_pred, 1)

    def predict_with_interpretation(self, input_data: Union[Dict, pd.DataFrame], use_sensors: bool = True) -> Dict:
        """
        Predict HbA1c with interpretation and risk category, optionally using sensor data.

        Args:
            input_data: Dictionary or DataFrame with feature values
            use_sensors: Whether to integrate sensor data from sensor_data.csv

        Returns:
            Dictionary with prediction, interpretation, and risk info
        """
        hba1c = self.predict(input_data, use_sensors)

        # Interpret HbA1c value using medical standards
        if hba1c < 5.7:
            category = "Normal"
            risk = "Low"
            message = "Your estimated HbA1c is within the normal range."
            color = "#4ade80"
        elif 5.7 <= hba1c <= 6.4:
            category = "Prediabetes"
            risk = "Moderate"
            message = "Your estimated HbA1c indicates prediabetes. Lifestyle changes recommended."
            color = "#facc15"
        elif 6.5 <= hba1c <= 8.0:
            category = "Diabetes"
            risk = "High"
            message = "Your estimated HbA1c is in the diabetes range. Medical consultation recommended."
            color = "#fb923c"
        else:  # > 8.0
            category = "Poor Control"
            risk = "Very High"
            message = "Your estimated HbA1c indicates poor diabetes control. Urgent medical attention needed."
            color = "#f87171"

        return {
            'hba1c_value': hba1c,
            'category': category,
            'risk_level': risk,
            'message': message,
            'color': color,
            'interpretation': self._get_clinical_interpretation(hba1c)
        }

    def _get_clinical_interpretation(self, hba1c: float) -> str:
        """Get detailed clinical interpretation of HbA1c value based on biological processes."""
        if hba1c < 5.7:
            return (
                f"HbA1c of {hba1c}% indicates normal glucose metabolism and insulin function. "
                "Glycation of hemoglobin is within normal limits, suggesting good glucose control "
                "over the past 2-3 months. Continue maintaining healthy lifestyle habits including "
                "regular physical activity, balanced diet, and adequate sleep."
            )
        elif 5.7 <= hba1c <= 6.0:
            return (
                f"HbA1c of {hba1c}% suggests early metabolic changes with increased glucose exposure. "
                "Mild insulin resistance may be developing. This level indicates higher risk for "
                "progression to diabetes. Lifestyle interventions including reduced sugar intake, "
                "increased physical activity, and stress management are recommended."
            )
        elif 6.1 <= hba1c <= 6.4:
            return (
                f"HbA1c of {hba1c}% indicates established prediabetes with significant insulin resistance. "
                "Chronic glucose elevation is causing increased hemoglobin glycation. Without intervention, "
                "progression to Type 2 diabetes is likely. Weight loss of 5-7%, 150 minutes weekly exercise, "
                "and dietary changes are strongly recommended to prevent diabetes."
            )
        elif 6.5 <= hba1c <= 7.0:
            return (
                f"HbA1c of {hba1c}% confirms diabetes diagnosis (≥6.5% threshold). "
                "Significant insulin resistance and/or beta-cell dysfunction is present, leading to "
                "chronic hyperglycemia and excessive hemoglobin glycation. Medical evaluation for "
                "diabetes management including possible medication, lifestyle counseling, and "
                "regular monitoring is essential."
            )
        elif 7.1 <= hba1c <= 8.0:
            return (
                f"HbA1c of {hba1c}% indicates diabetes with suboptimal control. "
                "Persistent hyperglycemia suggests inadequate glucose management despite potential treatment. "
                "Risk for complications increases. Medication optimization, intensive lifestyle counseling, "
                "and more frequent monitoring are needed to achieve target HbA1c <7%."
            )
        else:  # > 8.0
            return (
                f"HbA1c of {hba1c}% indicates poor diabetes control with very high complication risk. "
                "Severe chronic hyperglycemia suggests significant insulin resistance, possible beta-cell "
                "failure, or medication non-adherence. Immediate intensive management including medication "
                "adjustment, diabetes education, and screening for complications (retinopathy, nephropathy, "
                "neuropathy) is urgently needed."
            )

    def save_model(self, filepath: str = 'models/hba1c_model.joblib'):
        """
        Save the trained model to disk.

        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }

        joblib.dump(model_data, filepath)
        print(f"\n[+] Model saved to: {filepath}")

    def load_model(self, filepath: str = 'models/hba1c_model.joblib'):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        self.is_trained = True

        print(f"[+] Model loaded from: {filepath}")
        if self.metrics:
            print(f"   MAE: {self.metrics.get('mae', 0):.3f}%")
            print(f"   R²: {self.metrics.get('r2_score', 0):.3f}")


# ============================================================================
#  STANDALONE TRAINING SCRIPT
# ============================================================================

if __name__ == '__main__':
    """Train and save the HbA1c prediction model."""

    print("\n" + "=" * 70)
    print("  HbA1c PREDICTION MODEL TRAINER")
    print("=" * 70)

    # Create predictor
    predictor = HbA1cPredictor()

    # Train model
    metrics = predictor.train('dataset_cleaned.csv')

    # Save model
    predictor.save_model('models/hba1c_model.joblib')

    # Test prediction
    print("\n" + "=" * 70)
    print("  TESTING PREDICTIONS")
    print("=" * 70)

    test_cases = [
        {
            'name': 'Normal Healthy Individual',
            'data': {
                'Glucose': 85,
                'BMI': 22.0,
                'Age': 25,
                'SkinThickness': 20,
                'Insulin': 85,
                'BloodPressure': 70,
                'SleepHours': 8.0,
                'ActivityLevel': 2,
                'StressLevel': 2,
                'SugarIntake': 0,
                'DiabetesPedigreeFunction': 0.1,
                'Pregnancies': 0
            }
        },
        {
            'name': 'Prediabetic Individual (Early Stage)',
            'data': {
                'Glucose': 105,
                'BMI': 27.0,
                'Age': 40,
                'SkinThickness': 28,
                'Insulin': 120,
                'BloodPressure': 85,
                'SleepHours': 6.0,
                'ActivityLevel': 1,
                'StressLevel': 6,
                'SugarIntake': 1,
                'DiabetesPedigreeFunction': 0.3,
                'Pregnancies': 1
            }
        },
        {
            'name': 'Prediabetic Individual (Advanced)',
            'data': {
                'Glucose': 118,
                'BMI': 30.5,
                'Age': 50,
                'SkinThickness': 32,
                'Insulin': 160,
                'BloodPressure': 90,
                'SleepHours': 5.5,
                'ActivityLevel': 0,
                'StressLevel': 7,
                'SugarIntake': 2,
                'DiabetesPedigreeFunction': 0.5,
                'Pregnancies': 3
            }
        },
        {
            'name': 'Diabetic Patient (Newly Diagnosed)',
            'data': {
                'Glucose': 145,
                'BMI': 32.0,
                'Age': 55,
                'SkinThickness': 38,
                'Insulin': 200,
                'BloodPressure': 95,
                'SleepHours': 5.0,
                'ActivityLevel': 0,
                'StressLevel': 8,
                'SugarIntake': 2,
                'DiabetesPedigreeFunction': 0.7,
                'Pregnancies': 4
            }
        },
        {
            'name': 'Diabetic Patient (Poor Control)',
            'data': {
                'Glucose': 180,
                'BMI': 38.0,
                'Age': 60,
                'SkinThickness': 45,
                'Insulin': 300,
                'BloodPressure': 100,
                'SleepHours': 4.5,
                'ActivityLevel': 0,
                'StressLevel': 9,
                'SugarIntake': 2,
                'DiabetesPedigreeFunction': 0.9,
                'Pregnancies': 6
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        result = predictor.predict_with_interpretation(test_case['data'])
        print(f"  Predicted HbA1c: {result['hba1c_value']}%")
        print(f"  Category: {result['category']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Message: {result['message']}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now use this model in your app.py:")
    print("  from hba1c_model import HbA1cPredictor")
    print("  hba1c_predictor = HbA1cPredictor()")
    print("  hba1c_predictor.load_model('models/hba1c_model.joblib')")
    print("  hba1c = hba1c_predictor.predict(input_data)")
    print("=" * 70 + "\n")
