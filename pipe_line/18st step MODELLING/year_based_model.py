import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
input_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/16th step KNN IMPUTATION/8102024/"
output_folder = r"/havakalite/tarik/PM2.5_article/pipe_line/18st step MODELLING/year_based_model/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics."""
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

def prepare_data_by_year(all_data, target):
    """Prepare data for training and testing using leave-one-year-out approach."""
    logging.info("Preparing data for leave-one-year-out training and testing.")

    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
    all_data['Year'] = all_data['date'].dt.year
    all_data['Month'] = all_data['date'].dt.month
    all_data['Day'] = all_data['date'].dt.day

    all_data = all_data.dropna()

    unique_years = sorted(all_data['Year'].unique())
    logging.info(f"Found data for years: {unique_years}")

    test_years_data = []
    all_train_features = []
    all_X_train_scaled = []
    all_y_train = []
    all_X_test_scaled = []
    all_y_test = []
    all_target_scalers = []

    for test_year in unique_years:
        test_mask = (all_data['Year'] == test_year)
        test_data = all_data[test_mask].copy()
        train_data = all_data[~test_mask].copy()

        if len(train_data) == 0 or len(test_data) == 0:
            logging.warning(f"Insufficient data for year {test_year}")
            continue

        # Preserve Year, Month, Day, and Covid columns for the output
        test_years_data.append(test_data.copy())

        # Drop Year, Month, Day, and Covid from features after splitting
        cols_to_drop = ['Year', 'Month', 'Day', 'COVID - 19', 'date']
        train_data = train_data.drop(columns=cols_to_drop, errors='ignore')
        test_data = test_data.drop(columns=cols_to_drop, errors='ignore')

        # Prepare features
        features = [col for col in train_data.columns if col != target]

        # Scale features
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_train = train_data[features]
        X_test = test_data[features]

        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        y_train = target_scaler.fit_transform(train_data[target].values.reshape(-1, 1)).ravel()
        y_test = target_scaler.transform(test_data[target].values.reshape(-1, 1)).ravel()

        # Store prepared data
        all_train_features.append(X_train)
        all_X_train_scaled.append(X_train_scaled)
        all_y_train.append(y_train)
        all_X_test_scaled.append(X_test_scaled)
        all_y_test.append(y_test)
        all_target_scalers.append(target_scaler)

    return all_train_features, all_X_train_scaled, all_y_train, all_X_test_scaled, all_y_test, all_target_scalers, test_years_data

def save_predictions(test_years_data, xgb_forecasts, output_folder):
    """Save predictions with year information to Excel."""
    logging.info("Saving predictions to Excel.")

    combined_predictions = pd.DataFrame()

    for test_data, forecast in zip(test_years_data, xgb_forecasts):
        test_data = test_data.copy()
        test_data["PM2.5_predicted"] = forecast.ravel()
        combined_predictions = pd.concat([combined_predictions, test_data], ignore_index=True)

    combined_predictions.to_excel(os.path.join(output_folder, "actual_vs_predicted.xlsx"), index=False)

def save_evaluation_metrics(test_years_data, output_folder, target, xgb_forecasts):
    """Save evaluation metrics by year."""
    logging.info("Saving evaluation metrics.")

    year_metrics = []

    for test_data, forecast in zip(test_years_data, xgb_forecasts):
        test_year = test_data['Year'].iloc[0]
        y_true = test_data[target].values
        y_pred = forecast.ravel()

        rmse, mape, mase, r2 = evaluate_model(y_true, y_pred)

        year_metrics.append(["RMSE", rmse, test_year])
        year_metrics.append(["MAPE", mape, test_year])
        year_metrics.append(["MASE", mase, test_year])
        year_metrics.append(["R2", r2, test_year])

    # Calculate overall metrics
    all_true = np.concatenate([test_data[target].values for test_data in test_years_data])
    all_pred = np.concatenate([forecast.ravel() for forecast in xgb_forecasts])

    overall_rmse, overall_mape, overall_mase, overall_r2 = evaluate_model(all_true, all_pred)

    year_metrics.append(["RMSE", overall_rmse, "Overall"])
    year_metrics.append(["MAPE", overall_mape, "Overall"])
    year_metrics.append(["MASE", overall_mase, "Overall"])
    year_metrics.append(["R2", overall_r2, "Overall"])

    evaluation_metrics_df = pd.DataFrame(
        year_metrics,
        columns=["Metric", "Value", "Year"]
    )

    with pd.ExcelWriter(os.path.join(output_folder, "evaluation_metrics_by_year.xlsx")) as writer:
        for metric in evaluation_metrics_df["Metric"].unique():
            metric_df = evaluation_metrics_df[evaluation_metrics_df["Metric"] == metric]
            metric_df.to_excel(writer, sheet_name=metric, index=False)

def save_models(models, test_years_data, output_folder):
    """Save trained models for each year."""
    logging.info("Saving trained models.")

    models_folder = os.path.join(output_folder, "trained_models")
    os.makedirs(models_folder, exist_ok=True)

    for model, test_data in zip(models, test_years_data):
        year = test_data['Year'].iloc[0]
        model_filename = f"year_model_{year}.json"
        model.save_model(os.path.join(models_folder, model_filename))

def save_feature_importance(feature_importances, output_folder):
    """Save feature importance to Excel."""
    logging.info("Saving feature importance to Excel.")

    combined_feature_importances = pd.concat(feature_importances, ignore_index=True)
    combined_feature_importances.to_excel(os.path.join(output_folder, 'feature_importances.xlsx'), index=False)

def main():
    try:
        logging.info("Starting PM2.5 modeling process with leave-one-year-out approach.")

        # Load datasets
        logging.info("Loading and preprocessing data...")
        all_data = load_and_combine_datasets(input_folder)

        # Define the target variable
        target = "PM2.5"

        # Prepare data using leave-one-year-out approach
        results = prepare_data_by_year(all_data, target)
        if not results:
            logging.error("No data available after preprocessing.")
            return

        train_features, X_train_scaled, y_train, X_test_scaled, y_test, target_scalers, test_years_data = results

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

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # Train models for each year
        logging.info("Training models for each year...")
        best_xgb_models = []
        xgb_forecasts = []
        all_feature_importances = []

        for i, (X_train, y_train_data, X_test, target_scaler) in enumerate(zip(X_train_scaled, y_train, X_test_scaled, target_scalers)):
            test_year = test_years_data[i]['Year'].iloc[0]
            logging.info(f"Training model for year {test_year}")

            best_model, best_params = train_xgb_model(X_train, y_train_data, param_grid, kf)
            best_xgb_models.append(best_model)

            forecast = best_model.predict(X_test)
            forecast = target_scaler.inverse_transform(forecast.reshape(-1, 1))
            xgb_forecasts.append(forecast)

            # Get feature importances
            feature_importances = best_model.feature_importances_
            feature_names = train_features[i].columns
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            })
            feature_importance_df['Year'] = test_year
            all_feature_importances.append(feature_importance_df)

        # Save results
        logging.info("Saving results...")
        save_models(best_xgb_models, test_years_data, output_folder)
        save_predictions(test_years_data, xgb_forecasts, output_folder)
        save_evaluation_metrics(test_years_data, output_folder, target, xgb_forecasts)
        save_feature_importance(all_feature_importances, output_folder)

        logging.info("Processing completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()