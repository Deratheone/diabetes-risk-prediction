# GlycoSense - Diabetes Risk Prediction System

An advanced web-based diabetes risk prediction system using machine learning, with optional hardware integration for glucose monitoring and sleep tracking.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key (Optional - for AI suggestions)
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

To get a Gemini API key:
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy it to your `.env` file

### 3. Run the Application
```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## 📁 Project Structure

```
diabetes-risk-prediction/
├── app.py                    # Main Flask API server
├── diabetes_predictor.py     # Main ML model (Risk Classification)
├── hba1c_model.py            # HbA1c Estimation Model (Regression)
├── glucose_reader.py         # Hardware glucose sensor integration
├── test_gemini.py            # Test Gemini API connection
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── dataset_cleaned.csv       # Training dataset for risk model
├── sensor_data.csv           # Simulated/Logged sensor data
├── frontend/                 # Web interface
│   └── index.html            # Single-page application
├── hardware integration/     # Hardware device scripts
│   └── sleep_lifestyle.py    # Sleep tracker integration
├── models/                   # Saved ML models (risk & hba1c)
└── outputs/                  # Generated visualizations
```

## 🎯 Features

### Core Features
- **Multi-step Risk Assessment**: Comprehensive questionnaire covering:
  - Basic demographics (age, gender, weight, height)
  - Clinical data (blood pressure, pregnancies)
  - Family history of diabetes
  - Lifestyle factors (sugar intake)
  - Current medications

- **Advanced ML Predictions**:
  - **Diabetes Risk Score**: Classification model predicting probability of diabetes.
  - **Estimated HbA1c**: Regression model estimating HbA1c levels based on risk profile and glucose.
  - **Future Risk Projection**: Probabilistic forecasting of risk over 1, 3, 5, and 10 years.

- **Interactive Visualizations**:
  - Dynamic Gauge Chart for risk score.
  - Color-coded risk levels.
  - Visual breakdown of contributing risk factors.

- **AI-Powered Recommendations**: Personalized suggestions using Google's Gemini API (optional).

### Hardware & Sensor Integration
- **Real-time Sensor Data**: Integrates data from `sensor_data.csv` (simulated or logged) into predictions.
- **Glucose Analyzer**: Support for Arduino-based urine glucose detection (RGB sensor).
- **Sleep Tracker**: Support for wearable sleep and activity monitoring.


### Web Interface
- Modern, responsive single-page application
- Multi-step wizard with progress tracking
- Real-time validation and feedback
- Hardware integration modals
- Detailed risk analysis and recommendations

## 🔧 API Endpoints

### Main Endpoints
- `GET /` - Serve the web interface
- `GET /health` - Health check
- `POST /api/predict` - Get diabetes risk prediction
- `GET /api/model-info` - Get model information
- `POST /api/risk-reduction-suggestions` - Get AI-powered recommendations
- `POST /api/read-hardware/glucose` - Read glucose from hardware sensor
- `POST /api/read-hardware/sleep` - Read sleep data from hardware tracker

## 🧪 Testing

### Test Gemini API Connection
```bash
python test_gemini.py
```

This will verify:
- API key is configured correctly
- Connection to Gemini API works
- Available models and fallback strategy

## 🎛️ Hardware Setup (Optional)

### Glucose Analyzer
- Requires Arduino with RGB sensor on COM5
- See `glucose_reader.py` for implementation details

### Sleep Tracker
- Requires Arduino with accelerometer and heart rate sensor on COM5
- See `hardware integration/sleep_lifestyle.py` for details

## 📊 Machine Learning Models

The system uses a dual-model approach:

### 1. Risk Classification Model
- **Type**: Ensemble Classifier (Random Forest + Gradient Boosting)
- **Goal**: Predicts the binary probability of having diabetes.
- **Key Features**: Glucose, BMI, Age, Pedigree Function, Blood Pressure.

### 2. HbA1c Regression Model (`hba1c_model.py`)
- **Type**: Random Forest Regressor
- **Goal**: Estimates specific HbA1c percentage.
- **Logic**: Combines self-reported data with sensor inputs. Includes a physiological reality check against fasting glucose levels to ensure medical accuracy.

### Data Integration
- **Sensor Fusion**: The system can blend manual inputs with data from `sensor_data.csv` (simulating IoT devices) to improve prediction accuracy.
- **Smart Imputation**: Uses clinically-grounded formulas to estimate missing values (e.g., Insulin based on BMI/Glucose) when lab data is unavailable.

## 🔐 Environment Variables

Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

See `.env.example` for template.

## 📝 Notes

- The ML model is trained on synthetic/anonymized data
- Hardware integration is optional - the system works without it
- Gemini API integration is optional - fallback suggestions are provided if unavailable
- This is a screening tool, not a diagnostic tool - always consult healthcare professionals

## 🛠️ Development

### File Purposes
- **app.py**: Flask backend server handling API requests, predictions, and sensor data integration.
- **diabetes_predictor.py**: Main classification model pipeline.
- **hba1c_model.py**: Dedicated regression model for HbA1c estimation.
- **glucose_reader.py**: Serial communication with Arduino glucose sensor.
- **frontend/index.html**: Complete web UI with dynamic risk visualizations.

### Adding New Features
1. Backend changes go in `app.py`
2. ML model changes go in `diabetes_predictor.py`
3. Frontend changes go in `frontend/index.html`

## 📜 License

This project is for educational and research purposes.

---

**Last Updated**: 2026-03-25
**Version**: 1.0
