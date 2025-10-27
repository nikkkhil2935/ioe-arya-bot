"""
Retrain XGBoost models with the latest version of XGBoost.
This script retrains both the vacancy and vehicle type prediction models.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading preprocessed data...")
df = pd.read_csv('preprocessed_parking_data.csv')
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create vacancy labels based on realistic parking patterns
# Slots are more likely to be vacant during:
# - Very early morning (0-6)
# - Late night (22-24)
# - Weekends
# - Short durations suggest high turnover (more vacant slots)
np.random.seed(42)

# Calculate vacancy probability based on features
vacancy_prob = np.zeros(len(df))

# Base probability
vacancy_prob[:] = 0.4

# Increase vacancy probability during off-peak hours
off_peak_hours = (df['Entry_Hour'] < 6) | (df['Entry_Hour'] > 22)
vacancy_prob[off_peak_hours] += 0.3

# Increase vacancy on weekends
vacancy_prob[df['Is_Weekend'] == 1] += 0.15

# Adjust based on hour bins (off-peak bins get higher vacancy)
vacancy_prob[df['Hour_Bin'].isin([0, 5])] += 0.2

# Normalize probabilities to [0, 1]
vacancy_prob = np.clip(vacancy_prob, 0, 1)

# Generate vacancy labels based on calculated probabilities
df['Vacancy'] = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in vacancy_prob])

print("\n" + "="*60)
print("TRAINING VACANCY MODEL")
print("="*60)

# Features for vacancy model (excluding Duration as mentioned in dashboard.py)
vacancy_features = ['Entry_Hour', 'DayOfWeek', 'Is_Weekend', 'Hour_Bin']
X_vacancy = df[vacancy_features]
y_vacancy = df['Vacancy']

print(f"Features used: {vacancy_features}")
print(f"Target distribution:\n{y_vacancy.value_counts()}")

# Split data
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vacancy, y_vacancy, test_size=0.2, random_state=42, stratify=y_vacancy
)

# Train vacancy model
print("\nTraining vacancy model...")
vacancy_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
vacancy_model.fit(X_train_v, y_train_v)

# Evaluate
y_pred_v = vacancy_model.predict(X_test_v)
accuracy_v = accuracy_score(y_test_v, y_pred_v)
print(f"✓ Vacancy Model Accuracy: {accuracy_v:.4f}")

# Save vacancy model
joblib.dump(vacancy_model, 'xgb_parking_vacancy_model.pkl')
print("✓ Vacancy model saved as 'xgb_parking_vacancy_model.pkl'")

print("\n" + "="*60)
print("TRAINING VEHICLE TYPE MODEL")
print("="*60)

# Features for vehicle type model (including Duration)
vehicle_features = ['Entry_Hour', 'Duration', 'DayOfWeek', 'Is_Weekend', 'Hour_Bin']
X_vehicle = df[vehicle_features]
y_vehicle = df['Type of Vehicle_Two Wheeler'].astype(int)  # Convert bool to int

print(f"Features used: {vehicle_features}")
print(f"Target distribution:\n{y_vehicle.value_counts()}")

# Split data
X_train_vt, X_test_vt, y_train_vt, y_test_vt = train_test_split(
    X_vehicle, y_vehicle, test_size=0.2, random_state=42, stratify=y_vehicle
)

# Train vehicle type model
print("\nTraining vehicle type model...")
vehicle_type_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
vehicle_type_model.fit(X_train_vt, y_train_vt)

# Evaluate
y_pred_vt = vehicle_type_model.predict(X_test_vt)
accuracy_vt = accuracy_score(y_test_vt, y_pred_vt)
print(f"✓ Vehicle Type Model Accuracy: {accuracy_vt:.4f}")

# Save vehicle type model
joblib.dump(vehicle_type_model, 'xgb_vehicle_type_model.pkl')
print("✓ Vehicle type model saved as 'xgb_vehicle_type_model.pkl'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print("\nBoth models have been successfully trained and saved.")
print("You can now run the dashboard with: streamlit run dashboard.py")
