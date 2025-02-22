## Regional PM<sub>2.5</sub> Predictor (RPP)

<div align="justify">
The RPP is a machine learning-based model that estimates fine particulate matter (PM<sub>2.5</sub>) concentrations in the air, primarily focused on Turkiye. By integrating satellite-derived Aerosol Optical Thickness, high-resolution meteorological data, and ground-based air quality measurements, the RPP provides accurate predictions for both monitored and unmonitored areas. This scalable tool supports informed decision-making for air quality management, public health research, and environmental monitoring, ultimately contributing to the development of effective policies to improve population health and environmental well-being.
</div>
<br>
<div align="center">
    <img src="https://github.com/user-attachments/assets/721ce791-a2ab-4749-8508-93e7e663aa04" alt="PM2.5 article schema design" style="width: 75%;">
</div>  

# 

### Dataset Explanation: 
#### 1. date_of_pandemic_daily.xlsx:
A sample dataset tracking daily COVID-19 pandemic events (lockdowns, restrictions) to account for their influence on PM2.5 concentrations in the air quality patterns.  
#### 2. example_aot_data.nc:
A NetCDF file containing satellite-derived Aerosol Optical Thickness (AOT) measurements used for estimating PM2.5 concentrations in the model.  
#### 3. example_meteorology.nc:
A NetCDF file containing high-resolution meteorological data that is incorporated into the RPP model to improve PM2.5 concentration estimates based on weather conditions.   
#### 4. example_monitoring_results.xlsx:
A dataset containing ground-based air quality monitoring results that provides observed PM2.5 concentration data for training and validating the model.  
#### 5. station_coordinates.xlsx:
A dataset containing geographical coordinates of air quality monitoring stations used to spatially integrate measurements with other environmental data sources.    

# 

### Pipe-line Explanation:  
#### 01th step ERA-5 and AOD DATA IMPORT TO SQL:  
Importing ERA-5 reanalysis meteorological data and Aerosol Optical Depth (AOD) data into an SQL database for efficient querying and processing.  
#### 02nd step ERA-5 and AOD DATA ARRANGING OF POSITION:  
Organizing spatial data by aligning the geographic positions of ERA-5 and AOD datasets to ensure consistency across grids and stations.  
#### 03rd step ERA-5 and AOD DATA TIME-ZONE EQUALIZER:  
Standardizing timestamps across datasets by converting them to the same time zone, ensuring accurate temporal alignment for further analysis.  
#### 04th step ERA-5 and AOD DATA EXPORT DATA:  
Exporting processed and cleaned ERA-5 and AOD datasets from the SQL database to Excel and NetCDF formats for further processing.  
#### 05th step ERA-5 and AOD DATA GET DAILY AVG:  
Calculating daily averages for each meteorological and AOD variable to simplify temporal resolution and reduce noise in the dataset.  
#### 06th step SIM DIVIDING DIFFERENT EXCEL SHEETS:  
Splitting large simulation datasets into separate Excel sheets based on variables, regions, or time periods for easier handling and analysis.  
#### 07th step SIM DIVIDING DIFFERENT EXCEL FILES:  
Dividing multiple Excel files into smaller, categorized datasets to streamline processing and enhance computational efficiency.  
#### 08th step SIM REMOVING TIME FROM DATE COLUMN:  
Cleaning date columns by removing unnecessary time components, ensuring uniform date formatting across datasets.  
#### 09th step MERGING ERA-5 AOD AND SIM DATA:  
Integrating ERA-5, AOD, and simulation datasets into a unified dataset based on matching spatial and temporal coordinates.  
#### 10th step DATE SEPARATION:  
Splitting datasets by date ranges (e.g., pre-pandemic, lockdown periods) to analyze temporal trends and their impact on PM2.5 concentrations.  
#### 11th step ADDING NEW FEATURES (RH, WS, AND PM RATIO):  
Creating new predictive features such as relative humidity (RH), wind speed (WS), and PM2.5 ratios to improve model accuracy.  
#### 12th step ADDING COORDINATE OF STATION:  
Appending geographical coordinates for each monitoring station to enable spatial analysis and model predictions across locations.  
#### 13th step MERGING ALL STATION IN ONE EXCEL:  
Consolidating data from all monitoring stations into a single Excel file for streamlined analysis and model training.  
#### 14th step ORDERED DATASET:  
Organizing the dataset by station, date, and feature variables to ensure consistency and facilitate efficient processing.  
#### 15th step ECDF PRUNING:  
Applying Empirical Cumulative Distribution Function (ECDF) pruning to remove extreme outliers and improve data quality.  
#### 16th step KNN IMPUTATION:  
Handling missing data using k-nearest neighbors (KNN) imputation, leveraging patterns in neighboring observations to fill gaps.  
#### 17th step STATISTICAL ANALYSIS:  
Conducting exploratory data analysis (EDA), including correlation matrices, distributions, and trend analysis, to identify key relationships.  
#### 18th step MODELLING:  
Training machine learning models (e.g., Random Forest, XGBoost) on the processed dataset, tuning hyperparameters, and evaluating model performance using cross-validation metrics.  
</div>
