# Smart Parking Dashboard 🚗

A machine learning-powered parking occupancy and vehicle type prediction system built with Streamlit and XGBoost.

## Features

- 🎯 **Vacancy Prediction**: Predicts whether a parking slot is vacant or occupied based on temporal features
- 🚙 **Vehicle Type Detection**: Classifies vehicles as Two-Wheeler or Four-Wheeler
- 📊 **Interactive Dashboard**: User-friendly Streamlit interface for real-time predictions
- 🤖 **ML Models**: XGBoost classifiers trained on parking data with engineered features

## Live Demo

[Add your Streamlit Cloud deployment link here]

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/nikkkhil2935/ioe-arya-bot.git
cd ioe-arya-bot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
streamlit run dashboard.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
ioe-arya-bot/
├── dashboard.py                      # Main Streamlit application
├── retrain_models.py                 # Script to retrain ML models
├── convert_models.py                 # Model conversion utilities
├── xgb_parking_vacancy_model.pkl     # Trained vacancy prediction model
├── xgb_vehicle_type_model.pkl        # Trained vehicle type model
├── preprocessed_parking_data.csv     # Preprocessed training data
├── requirements.txt                  # Python dependencies
├── Procfile                          # Heroku deployment config
└── README.md                         # This file
```

## Models

### Vacancy Prediction Model
- **Features**: Entry Hour, Day of Week, Is Weekend, Hour Bin
- **Algorithm**: XGBoost Classifier
- **Accuracy**: ~68.6%

### Vehicle Type Prediction Model
- **Features**: Entry Hour, Duration, Day of Week, Is Weekend, Hour Bin
- **Algorithm**: XGBoost Classifier
- **Accuracy**: ~50.7%

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository: `nikkkhil2935/ioe-arya-bot`
6. Set main file path: `dashboard.py`
7. Click "Deploy"

## Retraining Models

To retrain the models with fresh data:

```bash
python retrain_models.py
```

This will:
1. Load preprocessed parking data
2. Create vacancy labels based on temporal patterns
3. Train both XGBoost models
4. Save updated model files

## Technologies Used

- **Python 3.12**
- **Streamlit** - Web dashboard framework
- **XGBoost** - Machine learning algorithm
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - ML utilities and metrics
- **Joblib** - Model serialization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Nikhil**
- GitHub: [@nikkkhil2935](https://github.com/nikkkhil2935)

## Acknowledgments

- Built as part of IOE Arya project
- Uses XGBoost for high-performance predictions
- Deployed on Streamlit Cloud for easy accessibility
