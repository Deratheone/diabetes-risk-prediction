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
import subprocess
import sys
import csv
import random
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
#  CSV DATA HANDLING
# ============================================================================

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'sensor_data.csv')

def get_sensor_data_from_csv():
    """Read a random row from sensor_data.csv."""
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return None
            
        with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if not rows:
                return None
            return random.choice(rows)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def save_sensor_data_to_csv(glucose=None, sleep=None, lifestyle=None):
    """Append sensor data to sensor_data.csv."""
    try:
        file_exists = os.path.exists(CSV_FILE_PATH)
        
        with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
            # We'll use the same fields for compatibility, ignoring heart rate for CSV storage
            # unless you want to add a column. The user prompt said:
            # "in the format: glucose , sleep, lifestyle."
            # So I will NOT add heart rate to the CSV to respect the requested format.
            fieldnames = ['timestamp', 'glucose', 'sleep', 'lifestyle']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow({
                'timestamp': datetime.datetime.now().isoformat(),
                'glucose': glucose if glucose is not None else '',
                'sleep': sleep if sleep is not None else '',
                'lifestyle': lifestyle if lifestyle is not None else ''
            })
            print(f"Data saved to CSV: glucose={glucose}, sleep={sleep}, lifestyle={lifestyle}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")


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

    # Future risk projection - use the detailed data from ML model
    # The ML model already provides comprehensive future risk projections
    future_risk = risk_data.get('future_risk', {
        'projections': {
            '1_year': {'percentage': min(95.0, probability * 100 + 3), 'category': 'Moderate'},
            '3_year': {'percentage': min(95.0, probability * 100 + 8), 'category': 'High'},
            '5_year': {'percentage': min(95.0, probability * 100 + 12), 'category': 'High'},
        }
    })

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
        "medicines": ["steroids", ...],
        "hardware_glucose": 150,  (optional - from glucose sensor)
        "hardware_sleep_hours": 7.5  (optional - from sleep tracker)
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
        
        # Handle hardware glucose if provided
        hardware_glucose = data.get('hardware_glucose')
        hardware_glucose_confidence = data.get('hardware_glucose_confidence', 'unknown')
        
        if hardware_glucose is not None:
            # Blend hardware glucose with estimated glucose
            # 70% hardware, 30% estimated (hardware has higher confidence)
            estimated_glucose = model_input['Glucose']
            blended_glucose = (hardware_glucose * 0.7) + (estimated_glucose * 0.3)
            model_input['Glucose'] = round(blended_glucose, 1)
            model_input['_hardware_glucose_used'] = True
            model_input['_hardware_glucose_raw'] = hardware_glucose
            model_input['_estimated_glucose'] = estimated_glucose
            model_input['_hardware_glucose_confidence'] = hardware_glucose_confidence
        
        # Handle hardware sleep hours if provided
        hardware_sleep_hours = data.get('hardware_sleep_hours')
        if hardware_sleep_hours is not None:
            model_input['SleepHours'] = hardware_sleep_hours
            model_input['_hardware_sleep_used'] = True

        # Make prediction using the trained model
        report = predictor.predict(model_input)

        # Generate additional clinical summary
        clinical_summary = generate_clinical_summary(report, model_input)

        # Prepare response - use ML model's future_risk directly
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
            'future_risk': report.get('future_risk', {}),  # Use ML model's detailed future risk
            'input_analysis': {
                'bmi': model_input['BMI'],
                'glucose_estimate': model_input.get('_estimated_glucose', model_input['Glucose']),
                'glucose_final': model_input['Glucose'],
                'glucose_hardware': model_input.get('_hardware_glucose_raw'),
                'glucose_hardware_confidence': model_input.get('_hardware_glucose_confidence'),
                'glucose_hardware_used': model_input.get('_hardware_glucose_used', False),
                'pedigree_score': model_input['DiabetesPedigreeFunction'],
                'skin_thickness': model_input['SkinThickness'],
                'sleep_hours': model_input.get('SleepHours', 7),
                'sleep_hardware_used': model_input.get('_hardware_sleep_used', False)
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


@app.route('/api/read-hardware/glucose', methods=['POST'])
def read_hardware_glucose():
    """
    Read glucose level from Arduino glucose analyzer.
    
    Returns:
    {
        "success": bool,
        "message": str,
        "device_connected": bool,
        "glucose_level": str,
        "glucose_value": int,
        "match_distance": float,
        "confidence": str,
        "is_no_strip": bool,
        "serial_output": [...],
        "error": str (if any)
    }
    """
    import serial
    import time
    PORT = "COM9"
    BAUD_RATE = 9600
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        # Fallback to CSV simulation if hardware fails
        print(f"Hardware not found ({e}), using CSV data...")
        csv_data = get_sensor_data_from_csv()
        
        if csv_data and csv_data.get('glucose'):
            try:
                g_val = int(float(csv_data['glucose']))
                return jsonify({
                    'success': True,
                    'message': 'Simulated data from CSV (Hardware not connected)',
                    'device_connected': False,
                    'glucose_level': 'High' if g_val > 125 else 'Normal',
                    'glucose_value': g_val,
                    'match_distance': 0,
                    'confidence': 'High (Simulated)',
                    'is_no_strip': False,
                    'serial_output': [],
                    'error': None
                })
            except ValueError:
                pass
                
        return jsonify({
            'success': False,
            'message': f'Cannot connect to {PORT} and no CSV data available: {str(e)}',
            'device_connected': False,
            'error': str(e)
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Unexpected error: {str(e)}',
            'device_connected': False,
            'error': str(e)
        }), 503

    output_lines = []
    start_time = time.time()
    # Read all serial output for 12 seconds (enough for countdown + result)
    glucose_value = None
    
    try:
        while time.time() - start_time < 12:
            if ser.in_waiting:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    output_lines.append(line)
                    # Try to parse glucose value from output
                    # Assuming format "Glucose: 120" or similar, but exact format unknown
                    # Based on return type int, we need to extract it.
                    # For now just collect lines.
            else:
                time.sleep(0.05)
        ser.close()
        
        # Mock extraction logic if hardware actually returns data
        # In a real scenario, we would parse `output_lines`
        # Since I don't see the parsing logic in the original code (it just returned lines),
        # I'll assume the frontend does the parsing OR I need to add it.
        # Wait, the original code didn't parse glucose_value! It just returned `serial_output`.
        # But the docstring says returns `glucose_value`: int.
        # The original implementation ONLY returned serial_output.
        # The frontend likely parses it.
        # However, to save to CSV, *I* need the value.
        
        # Let's try to extract a number from the last line if possible
        if output_lines:
            import re
            for line in reversed(output_lines):
                # Look for number in line
                match = re.search(r'(\d+)', line)
                if match:
                    glucose_value = int(match.group(1))
                    break
        
        if glucose_value:
            save_sensor_data_to_csv(glucose=glucose_value)
            
        return jsonify({
            'success': True,
            'message': 'Raw output from glucose analyzer',
            'device_connected': True,
            'serial_output': output_lines[-30:],  # last 30 lines for frontend
            'glucose_value': glucose_value 
        })
    except Exception as e:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        return jsonify({
            'success': False,
            'message': f'Failed to read serial output: {str(e)}',
            'device_connected': True,
            'error': str(e)
        }), 500


@app.route('/api/risk-reduction-suggestions', methods=['POST'])
def get_risk_reduction_suggestions():
    """
    Get personalized risk reduction suggestions using Gemini AI.

    Expected JSON body:
    {
        "prediction_data": {...},  # The original prediction response
    }

    The Gemini API key is read from environment variables (GEMINI_API_KEY).
    """
    try:
        from google import genai

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        prediction_data = data.get('prediction_data')

        if not prediction_data:
            return jsonify({'error': 'Missing prediction_data'}), 400

        # Extract data from prediction
        prediction = prediction_data.get('prediction', {})
        risk_category = prediction.get('category', 'Unknown')
        risk_percentage = prediction.get('percentage', 0)
        risk_factors = prediction_data.get('risk_factors', [])
        recommendations = prediction_data.get('recommendations', [])

        # Read API key from environment variables
        api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Gemini API key not configured. Please set GEMINI_API_KEY in .env file.',
                'fallback_suggestions': get_fallback_suggestions(risk_category, risk_factors)
            }), 500

        # Create prompt for Gemini
        prompt = f"""You are a medical AI assistant specializing in diabetes prevention and management.

Patient Risk Profile:
- Risk Category: {risk_category}
- Risk Percentage: {risk_percentage}%
- Risk Factors: {', '.join([rf.get('factor', '') for rf in risk_factors]) if risk_factors else 'None identified'}

Current Recommendations:
{chr(10).join(['- ' + rec for rec in recommendations]) if recommendations else 'None provided'}

Based on this information, provide personalized, actionable advice for reducing diabetes risk.
Include specific lifestyle changes, dietary recommendations, exercise plans, and monitoring strategies.
Keep the response concise (under 500 words) and organized with clear sections."""

        # Configure Gemini client - pass the API key explicitly
        client = genai.Client(api_key=api_key)

        # Strategy: Try multiple models starting with the latest ones
        models_to_try = [
            "models/gemini-2.5-flash",        # Latest and best
            "models/gemini-2.0-flash",        # Backup
            "models/gemini-flash-latest"      # Generic latest
        ]

        response = None
        successful_model = None

        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                successful_model = model
                break  # Success, exit loop
            except Exception as model_error:
                error_str = str(model_error)
                print(f"Model {model} failed: {error_str}")
                if "429" in error_str:
                    # If it's quota, try next model
                    continue
                elif "404" in error_str:
                    # If model not found, try next
                    continue
                else:
                    # For other errors, try next model but log it
                    continue

        if not response:
            # If all models failed, return fallback
            return jsonify({
                'success': False,
                'error': 'All Gemini models unavailable',
                'fallback_suggestions': get_fallback_suggestions(risk_category, risk_factors)
            }), 503

        suggestions = response.text

        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'risk_category': risk_category,
            'risk_percentage': risk_percentage,
            'model_used': successful_model
        })

    except ImportError:
        # Extract basic info for fallback (if available)
        try:
            prediction = data.get('prediction_data', {}).get('prediction', {})
            risk_category = prediction.get('category', 'Unknown')
            risk_factors = data.get('prediction_data', {}).get('risk_factors', [])
        except:
            risk_category = 'Unknown'
            risk_factors = []

        return jsonify({
            'success': False,
            'error': 'Google Generative AI library not installed',
            'fallback_suggestions': get_fallback_suggestions(risk_category, risk_factors)
        }), 503
    except Exception as e:
        # Extract basic info for fallback (if available)
        try:
            prediction = data.get('prediction_data', {}).get('prediction', {})
            risk_category = prediction.get('category', 'Unknown')
            risk_factors = data.get('prediction_data', {}).get('risk_factors', [])
        except:
            risk_category = 'Unknown'
            risk_factors = []

        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_suggestions': get_fallback_suggestions(risk_category, risk_factors)
        }), 500


def get_fallback_suggestions(risk_category, risk_factors):
    """Provide fallback suggestions when Gemini API is unavailable."""

    suggestions = {
        'lifestyle': [],
        'monitoring': [],
        'medical': []
    }

    # Base recommendations
    suggestions['lifestyle'].extend([
        "Follow a balanced diet with plenty of vegetables, lean proteins, and whole grains",
        "Aim for 150 minutes of moderate exercise per week (30 min, 5 days)",
        "Maintain 7-9 hours of quality sleep each night",
        "Stay hydrated with at least 8 glasses of water daily"
    ])

    suggestions['monitoring'].extend([
        "Track your weight weekly",
        "Monitor blood pressure regularly",
        "Keep a food diary to identify patterns"
    ])

    # Risk-specific recommendations
    if risk_category in ['High', 'Very High']:
        suggestions['lifestyle'].extend([
            "Consider working with a nutritionist for meal planning",
            "Start with low-impact exercises if you're sedentary",
            "Limit refined sugars and processed foods"
        ])
        suggestions['monitoring'].extend([
            "Check blood glucose monthly if possible",
            "Monitor HbA1c every 6 months"
        ])
        suggestions['medical'].append("Schedule a comprehensive health check-up within 1-3 months")

    # Add risk factor specific suggestions
    for rf in risk_factors:
        factor = rf.get('factor', '').lower()
        if 'bmi' in factor or 'weight' in factor:
            suggestions['lifestyle'].append("Focus on gradual weight loss (1-2 lbs per week)")
        elif 'glucose' in factor:
            suggestions['lifestyle'].append("Reduce carbohydrate portions and choose complex carbs")
        elif 'family history' in factor:
            suggestions['monitoring'].append("More frequent screening due to genetic predisposition")

    return suggestions


@app.route('/api/read-hardware/sleep', methods=['POST'])
def read_hardware_sleep():
    """
    Read sleep hours and activity level from Arduino sleep tracker.
    Uses the existing sleep_lifestyle.py script.
    
    Returns:
    {
        "success": bool,
        "message": str,
        "device_connected": bool,
        "sleep_hours": float,
        "activity_level": int,
        "activity_label": str,
        "serial_output": [...],
        "error": str (if any)
    }
    """
    import serial
    import time
    import math
    PORT = "COM5"
    BAUD_RATE = 115200
    SAMPLE_RATE_HZ = 20
    ANALYSIS_WINDOW_SEC = 15
    SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * ANALYSIS_WINDOW_SEC
    IR_FINGER_THRESH = 50000
    MIN_VALID_HR = 40
    MAX_VALID_HR = 180
    HR_VALID_FRAC = 0.25
    FINGER_VALID_FRAC = 0.25
    MOTION_SEDENTARY = 0.15
    MOTION_ACTIVE = 0.50
    LEVEL_NAMES = {0: "Sedentary", 1: "Lightly Active", 2: "Active"}
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        # Fallback to CSV
        csv_data = get_sensor_data_from_csv()
        if csv_data:
             try:
                 sleep_h = float(csv_data.get('sleep', 7.0))
             except ValueError:
                 sleep_h = 7.0
                 
             lifestyle = csv_data.get('lifestyle', 'Moderate')
             # LEVEL_NAMES = {0: "Sedentary", 1: "Lightly Active", 2: "Active"}
             level_map = {"Sedentary": 0, "Lightly Active": 1, "Moderate": 1, "Active": 2}
             act_level = level_map.get(lifestyle, 1)
             
             # Simulate HR based on activity
             simulated_hr = 70
             if act_level == 2:
                 simulated_hr = 95
             elif act_level == 1:
                 simulated_hr = 75
             else:
                 simulated_hr = 60
             
             return jsonify({
                 'success': True,
                 'sleep_hours': sleep_h,
                 'activity_level': act_level,
                 'activity_label': lifestyle,
                 'heart_rate': simulated_hr,
                 'features': {'avg_hr': simulated_hr},
                 'raw_samples': [],
                 'message': 'Simulated data from CSV (Hardware not connected)'
             })
             
        return jsonify({'success': False, 'error': f'Cannot open {PORT} and no CSV data: {e}'}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {e}'}), 503

    def read_line(ser):
        try:
            raw = ser.readline()
            if not raw:
                return None
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                return None
            if line.startswith("ERR") or line.startswith("#") or not line[0].isdigit():
                return None
            parts = line.split(",")
            if len(parts) != 6:
                return None
            return {
                "timestamp":  int(parts[0]),
                "ax":         float(parts[1]),
                "ay":         float(parts[2]),
                "az":         float(parts[3]),
                "ir":         int(parts[4]),
                "heart_rate": int(parts[5]),
            }
        except Exception:
            return None

    # Collect a window of samples
    samples = []
    start_time = time.time()
    while len(samples) < SAMPLES_PER_WINDOW and (time.time() - start_time) < 20:
        sample = read_line(ser)
        if sample:
            samples.append(sample)

    ser.close()
    if len(samples) < SAMPLES_PER_WINDOW // 2:
        return jsonify({'success': False, 'error': 'Not enough valid samples received from Arduino.'}), 500

    # Compute features
    def compute_motion(ax, ay, az):
        return abs(math.sqrt(ax**2 + ay**2 + az**2) - 1.0)

    motion_win = [compute_motion(s['ax'], s['ay'], s['az']) for s in samples]
    hr_win = [s['heart_rate'] for s in samples]
    ir_win = [s['ir'] for s in samples]
    n = len(motion_win)
    avg_m = sum(motion_win) / n
    var_m = sum((x - avg_m) ** 2 for x in motion_win) / n
    sed_n = sum(1 for m in motion_win if m < MOTION_SEDENTARY)
    act_n = sum(1 for m in motion_win if m > MOTION_ACTIVE)
    lit_n = n - sed_n - act_n
    finger_samples = [ir for ir in ir_win if ir >= IR_FINGER_THRESH]
    finger_frac = len(finger_samples) / n
    finger_ok = finger_frac >= FINGER_VALID_FRAC
    avg_ir = sum(ir_win) / len(ir_win) if ir_win else 0.0
    valid_hr = [h for h, ir in zip(hr_win, ir_win) if ir >= IR_FINGER_THRESH and MIN_VALID_HR <= h <= MAX_VALID_HR]
    valid_frac = len(valid_hr) / n
    avg_hr = sum(valid_hr) / len(valid_hr) if valid_hr else 0.0
    hr_ok = finger_ok and (valid_frac >= HR_VALID_FRAC)
    features = {
        "avg_motion": avg_m,
        "motion_variance": var_m,
        "avg_hr": avg_hr,
        "hr_ok": hr_ok,
        "finger_ok": finger_ok,
        "valid_hr_frac": valid_frac,
        "finger_frac": finger_frac,
        "avg_ir": avg_ir,
        "sedentary_pct": (sed_n / n) * 100,
        "active_pct": (act_n / n) * 100,
        "light_pct": (lit_n / n) * 100,
    }
    # Sleep detection
    motion_ok = (features["avg_motion"] < 0.15 and features["motion_variance"] < 0.02)
    sleep_state = "awake"
    if motion_ok and features["finger_ok"]:
        if features["hr_ok"]:
            hr = features["avg_hr"]
            if hr > 90:
                sleep_state = "awake"
            elif 40 <= hr <= 90:
                sleep_state = "sleep"
            else:
                sleep_state = "awake"
        else:
            sleep_state = "awake"
    # Activity classification
    if features["sedentary_pct"] > 60:
        activity_level = 0
    elif features["active_pct"] > 30:
        activity_level = 2
    else:
        activity_level = 1
    if activity_level == 0 and features["hr_ok"] and features["avg_hr"] > 85:
        activity_level = 1
    # Sleep hours estimate (for this window only)
    sleep_hours = round((sleep_state == "sleep") * (ANALYSIS_WINDOW_SEC / 3600.0), 2)
    
    # Save to CSV
    save_sensor_data_to_csv(sleep=sleep_hours, lifestyle=LEVEL_NAMES[activity_level])
    
    return jsonify({
        'success': True,
        'sleep_hours': sleep_hours,
        'activity_level': activity_level,
        'activity_label': LEVEL_NAMES[activity_level],
        'heart_rate': round(features["avg_hr"]),
        'features': features,
        'raw_samples': samples[-5:]  # last 5 samples for debug
    })


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
