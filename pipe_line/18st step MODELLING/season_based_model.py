import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
input_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/16th step KNN IMPUTATION/8102024/"
output_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/18st step MODELLING/seasonality_based_model/"
station_file = r"/havakalite/tarik/PM2.5_article/datasets/station_data/station_coordinates.xlsx"

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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Calculate MASE
    differencing = np.abs(np.diff(y_true))
    scale = np.mean(differencing) if np.mean(differencing) != 0 else 1  # Avoid division by zero
    errors = np.abs(y_true - y_pred)
    mase = np.mean(errors) / scale

    r2 = r2_score(y_true, y_pred)

    return rmse, mape, mase, r2

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

def train_xgb_model(X_train, y_train):
    """Train XGBoost model."""
    logging.info("Training XGBoost model.")

    xgb_model = xgb.XGBRegressor(
        n_estimators=5000,
        max_depth=9,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        gamma=1,
        max_bin=512,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    logging.info("Model training completed.")

    return xgb_model

def prepare_data_seasonality(all_data, target):
    """Prepare data for training and testing based on seasonality."""
    logging.info("Preparing data based on seasonality.")

    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
    all_data['Month'] = all_data['date'].dt.month

    # Define seasons
    season_mapping = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    all_data['Season'] = all_data['Month'].map(season_mapping)

    features = [col for col in all_data.columns if col not in [target, "date", "Season"]]
    all_data = all_data.dropna()

    # Split data into training and testing sets based on seasons
    test_data = pd.DataFrame()
    train_data = pd.DataFrame()

    for season in all_data['Season'].unique():
        season_data = all_data[all_data['Season'] == season]
        season_test = season_data.sample(frac=0.2, random_state=42)
        season_train = season_data.drop(season_test.index)

        test_data = pd.concat([test_data, season_test])
        train_data = pd.concat([train_data, season_train])

    logging.info(f"Training data records: {len(train_data)}, Testing data records: {len(test_data)}")

    # Prepare target variables
    y_train = train_data[target].values
    y_test = test_data[target].values

    # Scale target variables
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Prepare and scale features
    X_train = train_data[features]
    X_test = test_data[features]

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, target_scaler, test_data, train_data, feature_scaler

def save_predictions(test_data, y_test, y_pred, station_df, output_folder):
    """Save predictions with station information to Excel."""
    logging.info("Saving predictions to Excel.")

    test_data = test_data.copy()
    test_data["PM2.5_actual"] = y_test
    test_data["PM2.5_predicted"] = y_pred

    # Merge with station information
    test_data = test_data.merge(
        station_df[['longitude', 'latitude', 'station']],
        left_on=['Longitude', 'Latitude'],
        right_on=['longitude', 'latitude'],
        how='left'
    )

    # Organize columns
    cols = ['station', 'date', 'Longitude', 'Latitude', 'PM2.5_actual', 'PM2.5_predicted'] + \
           [col for col in test_data.columns if col not in ['station', 'date', 'Longitude', 'Latitude', 'PM2.5_actual', 'PM2.5_predicted', 'longitude', 'latitude']]
    test_data = test_data[cols]
    test_data.to_excel(os.path.join(output_folder, "actual_vs_predicted.xlsx"), index=False)

def save_feature_importances(model, X_train_columns, output_folder):
    """Save feature importances."""
    logging.info("Saving feature importances.")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": X_train_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    feature_importance_df.to_excel(os.path.join(output_folder, "feature_importances.xlsx"), index=False)

def save_evaluation_metrics(results_df, output_folder):
    """Save evaluation metrics by season."""
    logging.info("Saving evaluation metrics.")

    evaluation_metrics = []

    seasons = results_df['Season'].unique()

    for season in seasons:
        season_df = results_df[results_df['Season'] == season]

        y_true_season = season_df['PM2.5_actual']
        y_pred_season = season_df['PM2.5_predicted']

        rmse, mape, mase, r2 = evaluate_model(y_true_season, y_pred_season)

        evaluation_metrics.append({
            "Season": season,
            "RMSE": rmse,
            "MAPE": mape * 100,  # Convert to percentage
            "MASE": mase,
            "R2": r2
        })

    evaluation_metrics_df = pd.DataFrame(evaluation_metrics)
    evaluation_metrics_df.to_excel(os.path.join(output_folder, "evaluation_metrics_by_season.xlsx"), index=False)

def main():
    try:
        logging.info("Starting PM2.5 modeling process based on seasonality.")

        # Load datasets
        logging.info("Loading and preprocessing data...")
        all_data = load_and_combine_datasets(input_folder)
        all_data = add_dummy_variables(all_data)

        # Load station information
        station_df = pd.read_excel(station_file)

        # Define the target variable
        target = "PM2.5"

        # Prepare data based on seasonality
        (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
         target_scaler, test_data, train_data, feature_scaler) = prepare_data_seasonality(all_data, target)

        # Train the model
        logging.info("Training the model...")
        model = train_xgb_model(X_train_scaled, y_train_scaled)

        # Generate predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test = test_data[target].values  # Original y_test in original scale

        # Save the predictions
        save_predictions(test_data, y_test, y_pred, station_df, output_folder)

        # Combine results into a DataFrame
        results_df = test_data.copy()
        results_df['PM2.5_actual'] = y_test
        results_df['PM2.5_predicted'] = y_pred

        # Save the feature importances
        X_train_columns = train_data.drop(columns=[target, 'date', 'Season']).columns
        save_feature_importances(model, X_train_columns, output_folder)

        # Save the evaluation metrics
        save_evaluation_metrics(results_df, output_folder)

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()