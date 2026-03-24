"""
================================================================================
  DIABETES PREDICTION API SERVER
  ==============================
  Flask backend connecting the frontend to the ML model

  Endpoints:
  - GET  /                  - Serves the frontend
  - GET  /health            - Health check
  - POST /api/predict       - Make diabetes risk prediction
  - GET  /api/model-info    - Get model information

  Run: python app.py
================================================================================
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ============================================================================
#  FLASK APP SETUP
# ============================================================================

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for API calls

# Global predictor instance (lazy loaded)
predictor = None
model_loaded = False
loading_in_progress = False


def load_model():
    """Load the trained model (lazy loading)."""
    global predictor, model_loaded, loading_in_progress

    if model_loaded or loading_in_progress:
        return model_loaded

    loading_in_progress = True

    print("\n" + "=" * 60)
    print("  Loading Diabetes Prediction Model...")
    print("=" * 60)

    try:
        # Try loading saved model first (faster and avoids memory issues)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'diabetes_model.joblib')
        if os.path.exists(model_path):
            print("  Loading from saved model...")
            from diabetes_predictor import DiabetesPredictor
            predictor = DiabetesPredictor()
            predictor.load_model(model_path)
            model_loaded = True
            print("\n  Model loaded from saved file!")
        else:
            # Fallback: train new model
            print("  No saved model found, training new model...")
            from diabetes_predictor import DiabetesPredictor
            predictor = DiabetesPredictor()
            predictor.train(use_lifestyle=False)
            model_loaded = True
            print("\n  Model trained successfully!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False
    finally:
        loading_in_progress = False

    return model_loaded


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def map_frontend_to_model(data: dict) -> dict:
    """
    Map frontend form data to model input features.

    Frontend provides:
    - gender, age, weight, height, waist
    - pregnancies, bp (blood pressure)
    - relatives (list), othersClose
    - sugar (Low/Moderate/High)
    - medicines (list)

    Model expects:
    - Pregnancies, Glucose, BloodPressure, SkinThickness
    - Insulin, BMI, DiabetesPedigreeFunction, Age
    """

    # Extract values with defaults
    age = float(data.get('age', 30))
    weight = float(data.get('weight', 70))
    height = float(data.get('height', 170))
    waist = float(data.get('waist', 85))
    bp = float(data.get('bp', 72))
    pregnancies = int(data.get('pregnancies', 0)) if data.get('gender') == 'female' else 0

    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m * height_m) if height_m > 0 else 25

    # Calculate SkinThickness (from frontend formula)
    skin_thickness = max(0, 0.5 * bmi + 0.1 * waist - 5)

    # Calculate Diabetes Pedigree Function from relatives
    pedigree = calculate_pedigree(data.get('relatives', []), data.get('othersClose'))

    # Map sugar intake: Low=0, Moderate=1, High=2
    sugar_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    sugar_intake = sugar_map.get(data.get('sugar', 'Moderate'), 1)

    # Estimate glucose based on risk factors (since not collected in form)
    # This is a rough estimate - in production, this should be actual lab values
    glucose = estimate_glucose(age, bmi, sugar_intake, pedigree, data.get('medicines', []))

    # Estimate insulin (HOMA-IR based approximation)
    insulin = estimate_insulin(glucose, bmi)

    # Activity level (not in form, use default)
    activity_level = 1  # Medium

    # Stress level (not in form, use default based on medicines)
    stress_level = 5 + len(data.get('medicines', []))  # Higher if on medications

    # Sleep hours (not in form, use default)
    sleep_hours = 7

    return {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': round(skin_thickness, 1),
        'Insulin': insulin,
        'BMI': round(bmi, 1),
        'DiabetesPedigreeFunction': pedigree,
        'Age': int(age),
        'SleepHours': sleep_hours,
        'ActivityLevel': activity_level,
        'StressLevel': min(10, stress_level),
        'SugarIntake': sugar_intake
    }


def calculate_pedigree(relatives: list, others_close: str) -> float:
    """Calculate diabetes pedigree function from family history."""
    scores = {
        'father': 0.5,
        'mother': 0.5,
        'patGF': 0.25,
        'patGM': 0.25,
        'matGF': 0.25,
        'matGM': 0.25,
    }

    total = 0.0
    for rel in relatives:
        if rel == 'others':
            total += 0.2 if others_close == 'yes' else 0.1
        elif rel in scores:
            total += scores[rel]

    return round(total, 2)


def estimate_glucose(age: float, bmi: float, sugar: int, pedigree: float, medicines: list) -> float:
    """
    Estimate fasting glucose based on risk factors.
    NOTE: This is an approximation. Real applications should use actual lab values.
    """
    base = 85  # Normal baseline

    # Age factor
    if age > 60:
        base += 15
    elif age > 45:
        base += 10
    elif age > 35:
        base += 5

    # BMI factor
    if bmi > 35:
        base += 20
    elif bmi > 30:
        base += 12
    elif bmi > 25:
        base += 6

    # Sugar intake factor
    base += sugar * 8

    # Family history factor
    base += pedigree * 10

    # Medications factor (some medications raise glucose)
    diabetogenic_meds = ['steroids', 'antipsych', 'antirej']
    for med in medicines:
        if med in diabetogenic_meds:
            base += 10

    # Add some variance
    return round(min(200, max(70, base)), 0)


def estimate_insulin(glucose: float, bmi: float) -> float:
    """Estimate insulin based on glucose and BMI (HOMA-IR approximation)."""
    # Higher BMI and glucose typically mean higher insulin (insulin resistance)
    base = 50

    if glucose > 126:
        base += 40
    elif glucose > 100:
        base += 20

    if bmi > 30:
        base += 30
    elif bmi > 25:
        base += 15

    return round(min(300, max(20, base)), 0)


def generate_clinical_summary(risk_data: dict, input_data: dict) -> dict:
    """Generate a comprehensive clinical summary."""

    probability = risk_data.get('risk_probability', 0)
    category = risk_data.get('risk_category', 'Unknown')

    # Determine risk factors
    risk_factors = []

    if input_data.get('Glucose', 100) >= 126:
        risk_factors.append({'factor': 'Elevated glucose (diabetic range)', 'severity': 'high'})
    elif input_data.get('Glucose', 100) >= 100:
        risk_factors.append({'factor': 'Pre-diabetic glucose levels', 'severity': 'moderate'})

    if input_data.get('BMI', 25) >= 30:
        risk_factors.append({'factor': 'Obesity (BMI ≥ 30)', 'severity': 'high'})
    elif input_data.get('BMI', 25) >= 25:
        risk_factors.append({'factor': 'Overweight (BMI 25-30)', 'severity': 'moderate'})

    if input_data.get('Age', 30) > 45:
        risk_factors.append({'factor': 'Age above 45', 'severity': 'moderate'})

    if input_data.get('DiabetesPedigreeFunction', 0) > 0.5:
        risk_factors.append({'factor': 'Strong family history', 'severity': 'high'})
    elif input_data.get('DiabetesPedigreeFunction', 0) > 0:
        risk_factors.append({'factor': 'Family history present', 'severity': 'moderate'})

    if input_data.get('BloodPressure', 72) > 90:
        risk_factors.append({'factor': 'Elevated blood pressure', 'severity': 'moderate'})

    if input_data.get('SugarIntake', 1) == 2:
        risk_factors.append({'factor': 'High sugar intake', 'severity': 'moderate'})

    # Generate recommendations
    recommendations = []

    if probability >= 0.25:
        recommendations.append('Schedule HbA1c testing every 6-12 months')

    if input_data.get('BMI', 25) >= 25:
        recommendations.append('Target 5-7% weight loss through diet and exercise')

    if input_data.get('SugarIntake', 1) >= 1:
        recommendations.append('Reduce refined carbohydrates and added sugars')

    recommendations.append('Maintain 150+ minutes of moderate exercise weekly')

    if probability >= 0.5:
        recommendations.append('Consult a healthcare provider for comprehensive evaluation')

    # Future risk projection
    future_risk = {
        '1_year': min(0.95, probability + 0.03),
        '3_year': min(0.95, probability + 0.08),
        '5_year': min(0.95, probability + 0.12),
    }

    return {
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'future_risk': future_risk
    }


# ============================================================================
#  API ROUTES
# ============================================================================

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory('frontend', 'index.html')


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


@app.route('/api/model-info')
def model_info():
    """Get model information."""
    # Lazy load model
    if not model_loaded:
        if not load_model():
            return jsonify({'error': 'Model not loaded'}), 503

    return jsonify({
        'model_name': predictor.trainer.best_model_name,
        'features': predictor.processor.feature_names,
        'metrics': {
            'accuracy': predictor.trainer.results[predictor.trainer.best_model_name]['accuracy'],
            'roc_auc': predictor.trainer.results[predictor.trainer.best_model_name]['roc_auc'],
            'precision': predictor.trainer.results[predictor.trainer.best_model_name]['precision'],
            'recall': predictor.trainer.results[predictor.trainer.best_model_name]['recall'],
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a diabetes risk prediction.

    Expected JSON body:
    {
        "gender": "male" | "female" | "other",
        "age": 35,
        "weight": 70,
        "height": 170,
        "waist": 85,
        "pregnancies": 0,
        "bp": 72,
        "relatives": ["father", "mother", ...],
        "othersClose": "yes" | "no" | null,
        "sugar": "Low" | "Moderate" | "High",
        "medicines": ["steroids", ...]
    }
    """
    # Lazy load model on first request
    if not model_loaded:
        if not load_model():
            return jsonify({'error': 'Model failed to load'}), 503

    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Map frontend data to model inputs
        model_input = map_frontend_to_model(data)

        # Make prediction using the trained model
        report = predictor.predict(model_input)

        # Generate additional clinical summary
        clinical_summary = generate_clinical_summary(report, model_input)

        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'probability': report['risk_probability'],
                'percentage': report['risk_percentage'],
                'category': report['risk_category'],
                'message': report['risk_message']
            },
            'clinical_summary': report.get('clinical_summary', ''),
            'risk_factors': clinical_summary['risk_factors'],
            'recommendations': clinical_summary['recommendations'],
            'future_risk': clinical_summary['future_risk'],
            'input_analysis': {
                'bmi': model_input['BMI'],
                'glucose_estimate': model_input['Glucose'],
                'pedigree_score': model_input['DiabetesPedigreeFunction'],
                'skin_thickness': model_input['SkinThickness']
            },
            'warning_level': report.get('warning_level', {}),
            'disclaimer': (
                'This assessment is for screening purposes only and does not '
                'constitute a medical diagnosis. Please consult a healthcare '
                'provider for clinical evaluation and personalized advice.'
            )
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    # Load the model on startup
    load_model()

    # Run the server
    print("\n" + "=" * 60)
    print("  Starting GlycoSense API Server")
    print("=" * 60)
    print("  Frontend: http://localhost:5000")
    print("  API:      http://localhost:5000/api/predict")
    print("  Health:   http://localhost:5000/health")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
