import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from joblib import dump

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
input_model_folder = "16th step KNN IMPUTATION/imputed_dataset/"
model_output_folder = "18st step MODELLING/base_model_output/"

# Create output folder if it doesn't exist
os.makedirs(model_output_folder, exist_ok=True)

def load_and_combine_datasets(input_folder):
    """Load and combine all datasets from the input folder."""
    logging.info(f"Loading datasets from {input_folder}.")
    all_data = pd.DataFrame()
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            all_data = pd.concat([all_data, df], ignore_index=True)
    
    logging.info(f"Loaded {len(all_data)} total records from {input_folder}.")
    return all_data

def train_xgb_model(X_train, y_train, param_grid, kf):
    """Train XGBoost model with grid search CV."""
    logging.info("Training XGBoost model with GridSearchCV.")
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def prepare_data(data, target):
    """Prepare data for training."""
    logging.info("Preparing data for training.")
    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=[target])
    
    # Drop columns that are not needed or cannot be used without dummy variables
    data.drop(columns=['PM10', 'date', 'COVID - 19', 'Year', 'Month', 'Day'], inplace=True, errors='ignore')
    
    # Select features
    features = [col for col in data.columns if col != target]
    
    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X = feature_scaler.fit_transform(data[features])
    y = target_scaler.fit_transform(data[target].values.reshape(-1, 1)).ravel()
    
    return X, y, feature_scaler, target_scaler, features

def save_model_and_scalers(model, feature_scaler, target_scaler, features, output_folder):
    """Save the trained model, scalers, and feature list."""
    logging.info("Saving model and scalers.")
    
    # Save the model
    model_path = os.path.join(output_folder, "base_model.joblib")
    dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save the scalers
    feature_scaler_path = os.path.join(output_folder, "feature_scaler.joblib")
    target_scaler_path = os.path.join(output_folder, "target_scaler.joblib")
    dump(feature_scaler, feature_scaler_path)
    dump(target_scaler, target_scaler_path)
    logging.info("Scalers saved successfully")
    
    # Save feature list
    features_path = os.path.join(output_folder, "feature_list.txt")
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    logging.info("Feature list saved successfully")

def main():
    try:
        logging.info("Starting PM2.5 model training process.")
        
        # Load and preprocess training data
        train_data = load_and_combine_datasets(input_model_folder)
        
        # Prepare training data
        target = "PM2.5"
        X_train, y_train, feature_scaler, target_scaler, features = prepare_data(train_data, target)
        
        # Define the parameter grid for XGBoost model training
        param_grid = {
            "n_estimators": [500, 1000, 2000, 5000],
            "max_depth": [5, 7, 9, 11],
            "learning_rate": [0.01, 0.1, 0.3], 
            "subsample": [0.7, 0.8, 0.9], 
            "colsample_bytree": [0.7, 0.8, 0.9], 
            "reg_alpha": [0.1, 0.5, 1], 
            "reg_lambda": [0.1, 0.5, 1], 
            "max_bin": [256, 512], 
            "gamma": [1, 10] 
        }
        # INFO - Best hyperparameters found: {'colsample_bytree': 0.7, 'gamma': 1, 'learning_rate': 0.01, 'max_bin': 512, 'max_depth': 9, 'n_estimators': 5000, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 0.7}

        # Set up a K-Fold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train the model
        model = train_xgb_model(X_train, y_train, param_grid, kf)
        
        # Save the model and associated components
        save_model_and_scalers(model, feature_scaler, target_scaler, features, model_output_folder)
        
        logging.info("Model training and saving process completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()