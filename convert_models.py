"""
Script to convert old XGBoost models to new format.
This script attempts to load the old models and save them in a compatible format.
"""
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to downgrade xgboost temporarily to load the models
import subprocess
import sys

def convert_models():
    print("Step 1: Temporarily downgrading XGBoost to load old models...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.3.3", "--user", "--quiet"])
    
    print("Step 2: Loading old models...")
    import joblib
    
    try:
        vacancy_model = joblib.load('xgb_parking_vacancy_model.pkl')
        print("✓ Vacancy model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading vacancy model: {e}")
        vacancy_model = None
    
    try:
        vehicle_type_model = joblib.load('xgb_vehicle_type_model.pkl')
        print("✓ Vehicle type model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading vehicle type model: {e}")
        vehicle_type_model = None
    
    if vacancy_model is None or vehicle_type_model is None:
        print("\nCould not load models. They might be corrupted.")
        return False
    
    # Save in XGBoost's native format
    print("\nStep 3: Saving models in XGBoost native format...")
    vacancy_model.save_model('xgb_parking_vacancy_model.json')
    vehicle_type_model.save_model('xgb_vehicle_type_model.json')
    print("✓ Models saved as .json files")
    
    print("\nStep 4: Upgrading XGBoost to latest version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xgboost", "--user", "--quiet"])
    
    print("\n✓ Conversion complete! Models are now in .json format.")
    print("The dashboard.py file needs to be updated to load .json files instead of .pkl files.")
    return True

if __name__ == "__main__":
    try:
        success = convert_models()
        if not success:
            print("\nAlternative: You may need to retrain the models with the new XGBoost version.")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nThe models appear to be incompatible. You'll need to retrain them.")
