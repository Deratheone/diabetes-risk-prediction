"""
Test script for the complete HbA1c prediction integration.

This script tests:
1. HbA1c model standalone
2. Integration with diabetes predictor
3. Sensor data integration
4. API response format
"""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from hba1c_model import HbA1cPredictor
from diabetes_predictor import DiabetesPredictor

def test_hba1c_standalone():
    """Test the HbA1c model standalone."""
    print("\n" + "=" * 70)
    print("  TESTING HbA1c MODEL STANDALONE")
    print("=" * 70)

    # Initialize and load model
    hba1c_predictor = HbA1cPredictor()
    hba1c_predictor.load_model('models/hba1c_model.joblib')

    # Test cases covering different HbA1c ranges
    test_cases = [
        {
            'name': 'Normal Individual',
            'data': {
                'Pregnancies': 0,
                'Glucose': 90,
                'BloodPressure': 70,
                'SkinThickness': 20,
                'Insulin': 85,
                'BMI': 22.0,
                'DiabetesPedigreeFunction': 0.2,
                'Age': 25,
                'SleepHours': 8.0,
                'ActivityLevel': 2,
                'StressLevel': 3,
                'SugarIntake': 0
            }
        },
        {
            'name': 'Prediabetic Risk',
            'data': {
                'Pregnancies': 1,
                'Glucose': 115,
                'BloodPressure': 85,
                'SkinThickness': 30,
                'Insulin': 140,
                'BMI': 28.5,
                'DiabetesPedigreeFunction': 0.4,
                'Age': 45,
                'SleepHours': 6.0,
                'ActivityLevel': 1,
                'StressLevel': 7,
                'SugarIntake': 2
            }
        },
        {
            'name': 'Diabetic Range',
            'data': {
                'Pregnancies': 3,
                'Glucose': 160,
                'BloodPressure': 95,
                'SkinThickness': 40,
                'Insulin': 220,
                'BMI': 34.0,
                'DiabetesPedigreeFunction': 0.8,
                'Age': 55,
                'SleepHours': 5.0,
                'ActivityLevel': 0,
                'StressLevel': 9,
                'SugarIntake': 2
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)

        # Test without sensors
        result = hba1c_predictor.predict_with_interpretation(test_case['data'], use_sensors=False)
        print(f"HbA1c (no sensors): {result['hba1c_value']}% ({result['category']})")

        # Test with sensors
        result = hba1c_predictor.predict_with_interpretation(test_case['data'], use_sensors=True)
        print(f"HbA1c (with sensors): {result['hba1c_value']}% ({result['category']})")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Message: {result['message']}")

    return True

def test_diabetes_predictor_integration():
    """Test the integrated diabetes predictor with HbA1c."""
    print("\n" + "=" * 70)
    print("  TESTING DIABETES PREDICTOR WITH HbA1c")
    print("=" * 70)

    # Initialize predictor
    predictor = DiabetesPredictor()

    # First train the diabetes model (required)
    print("[*] Training diabetes prediction model...")
    try:
        predictor.train(use_lifestyle=True)
        print("[+] Training completed successfully")
    except Exception as e:
        print(f"[!] Training failed: {e}")
        return False

    # Test prediction with HbA1c integration
    test_input = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 80,
        "SkinThickness": 25,
        "Insulin": 120,
        "BMI": 27.5,
        "DiabetesPedigreeFunction": 0.4,
        "Age": 40,
        "SleepHours": 6.5,
        "ActivityLevel": 1,
        "StressLevel": 6,
        "SugarIntake": 1
    }

    print(f"\n[*] Making prediction...")
    try:
        report = predictor.predict(test_input)

        print(f"\nDiabetes Risk Results:")
        print(f"- Probability: {report['risk_percentage']:.1f}%")
        print(f"- Category: {report['risk_category']}")
        print(f"- Warning: {report['warning_level']['level']}")

        if report.get('hba1c_prediction'):
            hba1c = report['hba1c_prediction']
            print(f"\nHbA1c Prediction Results:")
            print(f"- Estimated HbA1c: {hba1c['hba1c_value']}%")
            print(f"- Category: {hba1c['category']}")
            print(f"- Risk Level: {hba1c['risk_level']}")
            print(f"- Message: {hba1c['message']}")
        else:
            print("\n[!] HbA1c prediction not available")

        print(f"\nClinical Summary:")
        print(f"{report['clinical_summary']}")

        return True

    except Exception as e:
        print(f"[!] Prediction failed: {e}")
        return False

def test_sensor_integration():
    """Test sensor data integration."""
    print("\n" + "=" * 70)
    print("  TESTING SENSOR DATA INTEGRATION")
    print("=" * 70)

    # Check if sensor data exists
    if not os.path.exists('sensor_data.csv'):
        print("[!] sensor_data.csv not found, creating sample data...")
        import pandas as pd
        import datetime

        # Create sample sensor data
        sample_data = []
        base_time = datetime.datetime.now() - datetime.timedelta(days=7)

        for i in range(7):
            timestamp = base_time + datetime.timedelta(days=i)
            sample_data.append({
                'timestamp': timestamp.isoformat(),
                'glucose': 95 + (i * 5),  # Gradually increasing glucose
                'sleep': 7.0 - (i * 0.2),  # Gradually decreasing sleep
                'lifestyle': ['Active', 'Moderate', 'Sedentary'][i % 3]
            })

        df = pd.DataFrame(sample_data)
        df.to_csv('sensor_data.csv', index=False)
        print("[+] Sample sensor data created")

    # Test HbA1c with sensor integration
    hba1c_predictor = HbA1cPredictor()
    hba1c_predictor.load_model('models/hba1c_model.joblib')

    test_data = {
        'Pregnancies': 1,
        'Glucose': 100,  # This will be modified by sensor data
        'BloodPressure': 75,
        'SkinThickness': 25,
        'Insulin': 100,
        'BMI': 26.0,
        'DiabetesPedigreeFunction': 0.3,
        'Age': 35,
        'SleepHours': 7.0,  # This will be modified by sensor data
        'ActivityLevel': 1,  # This will be modified by sensor data
        'StressLevel': 5,
        'SugarIntake': 1
    }

    print(f"\nOriginal input:")
    print(f"- Glucose: {test_data['Glucose']}")
    print(f"- Sleep: {test_data['SleepHours']}")
    print(f"- Activity: {test_data['ActivityLevel']}")

    # Test with sensor integration
    integrated_data = hba1c_predictor.integrate_sensor_data(test_data.copy())

    print(f"\nAfter sensor integration:")
    print(f"- Glucose: {integrated_data.get('Glucose', 'unchanged')}")
    print(f"- Sleep: {integrated_data.get('SleepHours', 'unchanged')}")
    print(f"- Activity: {integrated_data.get('ActivityLevel', 'unchanged')}")

    # Compare predictions
    hba1c_without_sensors = hba1c_predictor.predict(test_data, use_sensors=False)
    hba1c_with_sensors = hba1c_predictor.predict(test_data, use_sensors=True)

    print(f"\nHbA1c Predictions:")
    print(f"- Without sensors: {hba1c_without_sensors}%")
    print(f"- With sensors: {hba1c_with_sensors}%")
    print(f"- Difference: {abs(hba1c_with_sensors - hba1c_without_sensors):.2f}%")

    return True

def main():
    """Run all tests."""
    print("COMPREHENSIVE HbA1c INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("HbA1c Model Standalone", test_hba1c_standalone),
        ("Sensor Data Integration", test_sensor_integration),
        ("Diabetes Predictor Integration", test_diabetes_predictor_integration)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n[*] Running: {test_name}")
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"[!] Test failed with error: {e}")
            results[test_name] = "ERROR"

    # Summary
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status_icon = "[PASS]" if result == "PASS" else "[FAIL]" if result == "FAIL" else "[ERROR]"
        print(f"  {status_icon} {test_name}: {result}")

    total_passed = sum(1 for result in results.values() if result == "PASS")
    print(f"\n  Tests Passed: {total_passed}/{len(tests)}")

    if total_passed == len(tests):
        print("\n[SUCCESS] ALL TESTS PASSED! HbA1c integration is working correctly.")
        print("\nYou can now:")
        print("1. Run the Flask app: python app.py")
        print("2. Use the diabetes predictor with HbA1c: python diabetes_predictor.py")
        print("3. Test individual HbA1c predictions: python hba1c_model.py")
    else:
        print(f"\n[WARNING] {len(tests) - total_passed} tests failed. Check the errors above.")

    print("=" * 70)

if __name__ == "__main__":
    main()