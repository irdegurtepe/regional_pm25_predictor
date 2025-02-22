import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

# Set up logging to track the progress and any potential issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths for input and output directories, and station file
input_folder = "16th step KNN IMPUTATION/imputed_dataset/"
output_folder = "18st step MODELLING/general/"
station_file = "station_coordinates.xlsx"

# Function to add and then drop dummy variables for Year, Month, Day, and COVID-19
def add_dummy_variables(data):
    logging.info("Adding dummy variables for Year, Month, Day, and COVID-19.")
    
    # Convert Year, Month, Day, and COVID-19 to dummy variables
    year_dummies = pd.get_dummies(data['Year'], prefix='Year', drop_first=False)
    month_dummies = pd.get_dummies(data['Month'], prefix='Month', drop_first=False)
    day_dummies = pd.get_dummies(data['Day'], prefix='Day', drop_first=False)
    covid_dummy = pd.get_dummies(data['COVID - 19'], prefix='COVID', drop_first=True)
    
    # Concatenate the dummy variables with the original data
    data = pd.concat([data, year_dummies, month_dummies, day_dummies, covid_dummy], axis=1)
    
    # Drop the original Year, Month, Day, and COVID-19 columns
    data.drop(columns=['Year', 'Month', 'Day', 'COVID - 19'], inplace=True)
    
    # Drop the dummy variable columns
    data.drop(columns=data.filter(regex='^(Year_|Month_|Day_|COVID_)').columns, inplace=True)
    
    return data

# Function to evaluate the model using various metrics
def evaluate_model(y_true, y_pred):
    logging.info("Evaluating model performance.")
    y_true = y_true[-len(y_pred):]  # Match lengths of y_true and y_pred
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Calculate MASE
    diff_y_true = np.abs(y_true - np.roll(y_true, 1))[1:]  # Shift and remove first element to match length
    mase = np.mean(np.abs(y_true[1:] - y_pred[1:])) / np.mean(diff_y_true)
    
    # Calculate R2 Score
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mape, mase, r2

# Function to load and combine datasets from the input folder
def load_and_combine_datasets(input_folder):
    logging.info("Loading dataset.")
    all_data = pd.DataFrame()
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

# Function to prepare the data for training and testing
def prepare_data(all_data, target):
    logging.info("Preparing data for training and testing.")
    
    # Ensure the 'date' column is in datetime format
    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
    
    # Extract Year from 'date' column
    all_data['Year'] = all_data['date'].dt.year
    
    features = [col for col in all_data.columns if col not in [target, "date"]]
    
    # Drop rows with NaN values
    all_data = all_data.dropna()

    # Split the data into train and test
    train, test = train_test_split(all_data, test_size=0.2, random_state=42) # 80% train, 20% test - change the test size 0.3 for RS30 calculation
    
    target_column_train = train[target].copy()
    target_column_test = test[target].copy()

    # Save indices before scaling
    test_indices = test.index
    
    # Scale the target column
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(target_column_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(target_column_test.values.reshape(-1, 1))
    
    original_train_features = train[features]
    original_test_features = test[features]
    
    # Now scale the features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(original_train_features)
    X_test_scaled = feature_scaler.transform(original_test_features)
    
    y_train = y_train_scaled.ravel()
    y_test = y_test_scaled.ravel()
    
    # Use the original indices to retrieve test data
    test_with_stations = test.loc[test_indices]
    
    return original_train_features, X_train_scaled, y_train, X_test_scaled, y_test, target_scaler, test_with_stations

# Function to train the XGBoost model with cross-validation and parameter tuning
def train_xgb_model(X_train_scaled, y_train, param_grid, kf):
    logging.info("Starting XGBoost model training.")
    
    xgb_model = xgb.XGBRegressor(tree_method='hist')
    
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=kf,  # Use KFold here
        scoring="neg_mean_squared_error", 
        verbose=10, 
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_params = grid_search.best_params_
    logging.info(f"Best hyperparameters found: {best_params}")

    best_xgb_model = xgb.XGBRegressor(**best_params)
    best_xgb_model.fit(X_train_scaled, y_train, verbose=True)
    
    return best_xgb_model, best_params

# Function to save the predictions to an Excel file
def save_predictions(test_with_stations, xgb_forecast, output_folder):
    logging.info("Saving predictions to Excel.")
    test_with_stations["PM2.5_predicted"] = xgb_forecast.ravel()
    test_with_stations.to_excel(os.path.join(output_folder, "actual_vs_predicted.xlsx"), index=False)

# Function to save the feature importances to an Excel file
def save_feature_importances(best_xgb_model, X_train, output_folder):
    logging.info("Saving feature importances to Excel.")
    importances = best_xgb_model.feature_importances_
    
    feature_importance_dict = {feature: importance for feature, importance in zip(X_train.columns, importances)}
    
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    feature_importance_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
    feature_importance_df.to_excel(os.path.join(output_folder, "feature_importances.xlsx"), index=False)

def save_evaluation_metrics(test_with_stations, station_df, output_folder, target):
    logging.info("Saving evaluation metrics by station.")

    station_metrics = []

    # Group by rounded Longitude and Latitude to avoid precision errors
    grouped = test_with_stations.groupby([test_with_stations['Longitude'].round(4), 
                                          test_with_stations['Latitude'].round(4)])

    for (longitude, latitude), station_data in grouped:
        y_true = station_data[target].values
        y_pred = station_data["PM2.5_predicted"].values
        
        # Log small sample sizes
        if len(y_true) < 5:
            logging.warning(f"Small sample size for station at coordinates ({longitude}, {latitude}): {len(y_true)} samples.")

        # Proceed only if the lengths of y_true and y_pred match
        if len(y_true) > 0 and len(y_pred) > 0 and len(y_true) == len(y_pred):
            rmse, mape, mase, r2 = evaluate_model(y_true, y_pred)

            # Find the corresponding station name from the station_df
            station_name_values = station_df.loc[
                (station_df['longitude'].round(4) == longitude) & 
                (station_df['latitude'].round(4) == latitude), 
                'station'
            ].values

            if len(station_name_values) == 0:
                logging.warning(f"No matching station found for coordinates: ({longitude}, {latitude})")
                continue  # Skip this iteration if no match is found

            station_name = station_name_values[0]

            station_metrics.append(["RMSE", rmse, station_name, longitude, latitude])
            station_metrics.append(["MAPE", mape, station_name, longitude, latitude])
            station_metrics.append(["MASE", mase, station_name, longitude, latitude])
            station_metrics.append(["R2", r2, station_name, longitude, latitude])
            
            # Log if R2 is negative
            if r2 < 0:
                logging.warning(f"Negative R2 for station {station_name} at coordinates ({longitude}, {latitude}): R2 = {r2}")
        
        else:
            logging.warning(f"Mismatch or empty data for station at coordinates ({longitude}, {latitude}).")

    # Calculate overall metrics for the entire dataset
    y_true_all = test_with_stations[target].values
    y_pred_all = test_with_stations["PM2.5_predicted"].values

    overall_rmse, overall_mape, overall_mase, overall_r2 = evaluate_model(y_true_all, y_pred_all)
    
    # Append the overall metrics to the station_metrics list
    station_metrics.append(["RMSE", overall_rmse, "Overall", None, None])
    station_metrics.append(["MAPE", overall_mape, "Overall", None, None])
    station_metrics.append(["MASE", overall_mase, "Overall", None, None])
    station_metrics.append(["R2", overall_r2, "Overall", None, None])

    # Convert to DataFrame
    evaluation_metrics_df = pd.DataFrame(
        station_metrics, 
        columns=["Metric", "Value", "Station", "Longitude", "Latitude"]
    )

    # Save each metric to a separate sheet in the Excel file
    with pd.ExcelWriter(os.path.join(output_folder, "evaluation_metrics_by_station.xlsx")) as writer:
        for metric in evaluation_metrics_df["Metric"].unique():
            metric_df = evaluation_metrics_df[evaluation_metrics_df["Metric"] == metric]
            metric_df.to_excel(writer, sheet_name=metric, index=False)

    logging.info("Evaluation metrics have been saved successfully.")


def main():
    # Load and combine datasets from the input folder
    all_data = load_and_combine_datasets(input_folder)
    
    # Add dummy variables for Year, Month, Day, and COVID-19
    all_data = add_dummy_variables(all_data)
    
    # Define the target variable
    target = "PM2.5"
    
    # Prepare data for training and testing
    train_features, X_train_scaled, y_train, X_test_scaled, y_test, target_scaler, test_with_stations = prepare_data(all_data, target)
    
    # Load the station information
    station_df = pd.read_excel(station_file)

    # Define the parameter grid for XGBoost model training
    param_grid = {
        "n_estimators": [5000], # 500, 1000, 2000, 5000
        "max_depth": [9], # 5, 7, 9, 11
        "learning_rate": [0.01], # 0.01, 0.1, 0.3
        "subsample": [0.7], # 0.7, 0.8, 0.9
        "colsample_bytree": [0.7], # 0.7, 0.8, 0.9
        "reg_alpha": [0.1], # 0.1, 0.5, 1
        "reg_lambda": [0.1], # 0.1, 0.5, 1
        "max_bin": [512], # 256, 512
        "gamma": [1] # 1, 10
    }
    # INFO - Best hyperparameters found: {'colsample_bytree': 0.7, 'gamma': 1, 'learning_rate': 0.01, 'max_bin': 512, 'max_depth': 9, 'n_estimators': 5000, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 0.7}
    
    # Set up a K-Fold cross-validator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Train the XGBoost model with cross-validation and parameter tuning
    best_xgb_model, best_params = train_xgb_model(X_train_scaled, y_train, param_grid, kf)
    
    # Generate predictions on the test set
    xgb_forecast = best_xgb_model.predict(X_test_scaled)
    xgb_forecast = target_scaler.inverse_transform(xgb_forecast.reshape(-1, 1))
    
    # Save the predictions to an Excel file
    save_predictions(test_with_stations, xgb_forecast, output_folder)
    
    # Save the feature importances to an Excel file
    save_feature_importances(best_xgb_model, train_features, output_folder)
    
    # Save the evaluation metrics by station to an Excel file
    save_evaluation_metrics(test_with_stations, station_df, output_folder, target)

if __name__ == "__main__":
    main()
