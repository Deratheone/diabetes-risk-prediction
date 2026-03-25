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
├── diabetes_predictor.py     # ML model training & prediction
├── glucose_reader.py         # Hardware glucose sensor integration
├── test_gemini.py           # Test Gemini API connection
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
├── .env.example             # Template for environment variables
├── dataset_cleaned.csv      # Training dataset
├── frontend/                # Web interface
│   └── index.html          # Single-page application
├── hardware integration/    # Hardware device scripts
│   └── sleep_lifestyle.py  # Sleep tracker integration
├── models/                  # Saved ML models
└── outputs/                 # Generated visualizations
```

## 🎯 Features

### Core Features
- **Multi-step Risk Assessment**: Comprehensive questionnaire covering:
  - Basic demographics (age, gender, weight, height)
  - Clinical data (blood pressure, pregnancies)
  - Family history of diabetes
  - Lifestyle factors (sugar intake)
  - Current medications

- **Machine Learning Prediction**: Uses ensemble models trained on clinical data to predict diabetes risk

- **AI-Powered Recommendations**: Personalized suggestions using Google's Gemini API (optional)

### Hardware Integration (Optional)
- **Glucose Analyzer**: Urine glucose detection via Arduino RGB sensor
- **Sleep Tracker**: Sleep hours and activity level monitoring from wearable devices

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

## 📊 Machine Learning Model

The system uses multiple ML models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- HistGradient Boosting

### Features Used:
- **Clinical**: Glucose, Blood Pressure, Insulin, BMI, Age, Pregnancies
- **Genetic**: Diabetes Pedigree Function (family history)
- **Lifestyle**: Sleep hours, Activity level, Stress level, Sugar intake

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
- **app.py**: Flask backend server handling API requests and predictions
- **diabetes_predictor.py**: Complete ML pipeline with training, evaluation, and prediction
- **glucose_reader.py**: Serial communication with Arduino glucose sensor
- **test_gemini.py**: Utility to test Gemini API configuration
- **frontend/index.html**: Complete web UI (single file with inline CSS/JS)

### Adding New Features
1. Backend changes go in `app.py`
2. ML model changes go in `diabetes_predictor.py`
3. Frontend changes go in `frontend/index.html`

## 📜 License

This project is for educational and research purposes.

---

**Last Updated**: 2026-03-25
**Version**: 1.0
