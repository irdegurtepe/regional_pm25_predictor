import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
input_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/16th step KNN IMPUTATION/8102024/"
output_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/18st step MODELLING/station_based_model/"
station_file = r"/havakalite/tarik/PM2.5_article/datasets/station_data/station_coordinates.xlsx"
Year = 2023

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def add_dummy_variables(data):
    """Add and then drop dummy variables for temporal features."""
    logging.info("Adding dummy variables for Year, Month, Day, and COVID-19.")
    
    year_dummies = pd.get_dummies(data['Year'], prefix='Year', drop_first=False)
    month_dummies = pd.get_dummies(data['Month'], prefix='Month', drop_first=False)
    day_dummies = pd.get_dummies(data['Day'], prefix='Day', drop_first=False)
    covid_dummy = pd.get_dummies(data['COVID - 19'], prefix='COVID', drop_first=True)
    
    data = pd.concat([data, year_dummies, month_dummies, day_dummies, covid_dummy], axis=1)
    data.drop(columns=['Year', 'Month', 'Day', 'COVID - 19'], inplace=True)
    data.drop(columns=data.filter(regex='^(Year_|Month_|Day_|COVID_)').columns, inplace=True)
    
    return data

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics."""
    y_true = y_true[-len(y_pred):]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Calculate MASE
    diff_y_true = np.abs(y_true - np.roll(y_true, 1))[1:]
    mase = np.mean(np.abs(y_true[1:] - y_pred[1:])) / np.mean(diff_y_true)
    
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mape * 100, mase, r2

def load_and_combine_datasets(input_folder):
    """Load and combine all datasets from the input folder."""
    logging.info("Loading datasets from input folder.")
    all_data = pd.DataFrame()
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            all_data = pd.concat([all_data, df], ignore_index=True)
    
    logging.info(f"Loaded {len(all_data)} total records.")
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
    
    return grid_search.best_estimator_, grid_search.best_params_

def prepare_data(all_data, target):
    """Prepare data for training and testing using leave-one-station-out approach."""
    logging.info("Preparing data for leave-one-station-out training and testing.")
    
    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
    features = [col for col in all_data.columns if col not in [target, "date"]]
    all_data = all_data.dropna()
    
    # Get unique stations
    unique_stations = all_data.groupby(['Longitude', 'Latitude']).size().reset_index()[['Longitude', 'Latitude']]
    logging.info(f"Found {len(unique_stations)} unique stations.")
    
    all_train_features = []
    all_X_train_scaled = []
    all_y_train = []
    all_X_test_scaled = []
    all_y_test = []
    all_target_scalers = []
    all_test_stations = []
    
    # For each station, use it as test set and all others as training set
    for _, test_station in unique_stations.iterrows():
        # Split data into test station and training stations
        test_mask = (
            (all_data['Longitude'] == test_station['Longitude']) & 
            (all_data['Latitude'] == test_station['Latitude'])
        )
        
        # Get test data (current station)
        test = all_data[test_mask].copy()
        
        # Get train data (all other stations)
        train = all_data[~test_mask].copy()
        
        if len(train) == 0 or len(test) == 0:
            logging.warning(f"Insufficient data for station at ({test_station['Longitude']}, {test_station['Latitude']})")
            continue
        
        # Prepare target variables
        target_column_train = train[target].copy()
        target_column_test = test[target].copy()
        test_indices = test.index
        
        # Scale target variables
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(target_column_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(target_column_test.values.reshape(-1, 1))
        
        # Prepare and scale features
        original_train_features = train[features]
        original_test_features = test[features]
        
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(original_train_features)
        X_test_scaled = feature_scaler.transform(original_test_features)
        
        # Store prepared data
        all_train_features.append(original_train_features)
        all_X_train_scaled.append(X_train_scaled)
        all_y_train.append(y_train_scaled.ravel())
        all_X_test_scaled.append(X_test_scaled)
        all_y_test.append(y_test_scaled.ravel())
        all_target_scalers.append(target_scaler)
        all_test_stations.append(test.loc[test_indices])
    
    return all_train_features, all_X_train_scaled, all_y_train, all_X_test_scaled, all_y_test, all_target_scalers, all_test_stations


def save_predictions(test_with_stations, xgb_forecast, station_df, output_folder):
    """Save predictions with station information to Excel."""
    logging.info("Saving predictions to Excel.")
    
    combined_predictions = pd.DataFrame()
    
    for i, test_data in enumerate(test_with_stations):
        test_data = test_data.copy()
        longitude = test_data['Longitude'].iloc[0]
        latitude = test_data['Latitude'].iloc[0]
        
        # Get station name
        station_name = station_df.loc[
            (station_df['longitude'].round(4) == round(longitude, 4)) & 
            (station_df['latitude'].round(4) == round(latitude, 4)), 
            'station'
        ].iloc[0]
        
        test_data["Station_Name"] = station_name
        test_data["PM2.5_predicted"] = xgb_forecast[i].ravel()
        combined_predictions = pd.concat([combined_predictions, test_data], ignore_index=True)
    
    # Organize columns
    cols = ['Station_Name', 'date', 'Longitude', 'Latitude', 'PM2.5', 'PM2.5_predicted'] + \
           [col for col in combined_predictions.columns if col not in ['Station_Name', 'date', 'Longitude', 'Latitude', 'PM2.5', 'PM2.5_predicted']]
    
    combined_predictions = combined_predictions[cols]
    combined_predictions.to_excel(os.path.join(output_folder, "actual_vs_predicted.xlsx"), index=False)

def save_feature_importances(best_xgb_models, X_train, output_folder):
    """Save average feature importances across all models."""
    logging.info("Saving feature importances.")
    
    all_importances = []
    for model, features in zip(best_xgb_models, X_train):
        importances = model.feature_importances_
        importance_dict = {feature: importance for feature, importance in zip(features.columns, importances)}
        all_importances.append(pd.DataFrame([importance_dict]))
    
    avg_importance = pd.concat(all_importances).mean()
    
    feature_importance_df = pd.DataFrame({
        "Feature": avg_importance.index,
        "Importance": avg_importance.values
    }).sort_values("Importance", ascending=False)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance_df.head(20), x="Importance", y="Feature")
    plt.title("Top 20 Feature Importances (Average Across All Stations)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "feature_importances.png"))
    plt.close()
    
    feature_importance_df.to_excel(os.path.join(output_folder, "feature_importances.xlsx"), index=False)

def save_evaluation_metrics(test_with_stations, station_df, output_folder, target, xgb_forecasts):
    """Save evaluation metrics by station and create performance plots."""
    logging.info("Saving evaluation metrics and performance plots.")
    
    station_metrics = []
    
    for i, (test_data, forecast) in enumerate(zip(test_with_stations, xgb_forecasts)):
        longitude = test_data['Longitude'].iloc[0]
        latitude = test_data['Latitude'].iloc[0]
        
        station_name = station_df.loc[
            (station_df['longitude'].round(4) == round(longitude, 4)) & 
            (station_df['latitude'].round(4) == round(latitude, 4)), 
            'station'
        ].iloc[0]
        
        y_true = test_data[target].values
        y_pred = forecast.ravel()
        
        rmse, mape, mase, r2 = evaluate_model(y_true, y_pred)
        
        station_metrics.append(["RMSE", rmse, station_name, longitude, latitude])
        station_metrics.append(["MAPE", mape, station_name, longitude, latitude])
        station_metrics.append(["MASE", mase, station_name, longitude, latitude])
        station_metrics.append(["R2", r2, station_name, longitude, latitude])
        
        # Create performance plot for each station
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        plt.title(f'Actual vs Predicted PM2.5 for Station: {station_name}')
        plt.savefig(os.path.join(output_folder, f'performance_plot_{station_name}.png'))
        plt.close()
    
    # Calculate and save overall metrics
    all_true = np.concatenate([test_data[target].values for test_data in test_with_stations])
    all_pred = np.concatenate([forecast.ravel() for forecast in xgb_forecasts])
    
    overall_rmse, overall_mape, overall_mase, overall_r2 = evaluate_model(all_true, all_pred)
    
    station_metrics.append(["RMSE", overall_rmse, "Overall", None, None])
    station_metrics.append(["MAPE", overall_mape, "Overall", None, None])
    station_metrics.append(["MASE", overall_mase, "Overall", None, None])
    station_metrics.append(["R2", overall_r2, "Overall", None, None])
    
    evaluation_metrics_df = pd.DataFrame(
        station_metrics, 
        columns=["Metric", "Value", "Station", "Longitude", "Latitude"]
    )
    
    with pd.ExcelWriter(os.path.join(output_folder, "evaluation_metrics_by_station.xlsx")) as writer:
        for metric in evaluation_metrics_df["Metric"].unique():
            metric_df = evaluation_metrics_df[evaluation_metrics_df["Metric"] == metric]
            metric_df.to_excel(writer, sheet_name=metric, index=False)

def save_models(models, stations, station_df, output_folder):
    """Save trained models for each station."""
    logging.info("Saving trained models.")
    
    models_folder = os.path.join(output_folder, "trained_models")
    os.makedirs(models_folder, exist_ok=True)
    
    for model, station_data in zip(models, stations):
        longitude = station_data['Longitude'].iloc[0]
        latitude = station_data['Latitude'].iloc[0]
        
        station_name = station_df.loc[
            (station_df['longitude'].round(4) == round(longitude, 4)) & 
            (station_df['latitude'].round(4) == round(latitude, 4)), 
            'station'
        ].iloc[0]
        
        model_filename = f"station_model_{station_name}.json"
        model.save_model(os.path.join(models_folder, model_filename))

def main():
    try:
        logging.info("Starting PM2.5 modeling process with leave-one-station-out approach.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Load datasets
        logging.info("Loading and preprocessing data...")
        all_data = load_and_combine_datasets(input_folder)
        all_data = add_dummy_variables(all_data)
        
        # Load station information
        station_df = pd.read_excel(station_file)
        
        # Define the target variable
        target = "PM2.5"
        
        # Prepare data using leave-one-station-out approach
        train_features, X_train_scaled, y_train, X_test_scaled, y_test, target_scalers, test_with_stations = prepare_data(all_data, target)
        
        # Define the parameter grid for XGBoost model training
        param_grid = {
            "n_estimators": [5000],
            "max_depth": [9],
            "learning_rate": [0.01],
            "subsample": [0.7],
            "colsample_bytree": [0.7],
            "reg_alpha": [0.1],
            "reg_lambda": [0.1],
            "max_bin": [512],
            "gamma": [1]
        }
        
        # Set up a K-Fold cross-validator
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Train models for each station
        logging.info("Training models for each station...")
        best_xgb_models = []
        xgb_forecasts = []
        
        for i, (X_train, y_train_data, X_test, target_scaler) in enumerate(zip(X_train_scaled, y_train, X_test_scaled, target_scalers)):
            logging.info(f"Training model for station {i+1}/{len(X_train_scaled)}")
            
            # Train the XGBoost model
            best_model, best_params = train_xgb_model(X_train, y_train_data, param_grid, kf)
            best_xgb_models.append(best_model)
            
            # Generate predictions
            forecast = best_model.predict(X_test)
            forecast = target_scaler.inverse_transform(forecast.reshape(-1, 1))
            xgb_forecasts.append(forecast)
        
        # Save all outputs
        logging.info("Saving results...")
        
        # Save the models
        save_models(best_xgb_models, test_with_stations, station_df, output_folder)
        
        # Save the predictions
        save_predictions(test_with_stations, xgb_forecasts, station_df, output_folder)
        
        # Save the feature importances
        save_feature_importances(best_xgb_models, train_features, output_folder)
        
        # Save the evaluation metrics and create performance plots
        save_evaluation_metrics(test_with_stations, station_df, output_folder, target, xgb_forecasts)
        
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()