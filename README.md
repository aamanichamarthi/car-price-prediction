# car-price-prediction

This project focuses on predicting car prices using various machine learning regression models. The analysis involves comprehensive data preprocessing, exploratory data analysis (EDA) through visualizations, feature engineering, and rigorous model evaluation.

## Project Overview

The goal of this project is to build and evaluate machine learning models capable of predicting car prices based on a given dataset. The Jupyter Notebook walks through the entire process, from initial data loading to model selection and performance evaluation.

## Key Features & Analysis

* **Data Loading and Initial Inspection**: Loading the `train.csv` dataset and understanding its basic structure, including column types and summary statistics.
* **Data Cleaning & Preprocessing**:
    * Handling of missing values in the 'Levy' column by imputation with the mean.
    * Standardizing text data (e.g., replacing 'rover' with 'land rover' in 'Manufacturer').
    * Cleaning and converting the 'Mileage' column from string (e.g., '100000km') to integer.
    * Filtering out outliers in the 'Price' column (e.g., prices above 400,000).
    * Dropping irrelevant columns such as 'ID', 'Doors', and 'Prod. year'.
    * Removing duplicate entries from the dataset.
* **Exploratory Data Analysis (EDA) & Visualization**:
    * Distribution of Production Years.
    * Scatter plot showing the relationship between Production Year and Price.
    * Count of Cars by Manufacturer, highlighting top manufacturers.
    * Distribution of Engine Volumes (after cleaning ' Turbo' suffix).
    * Boxplot showing the distribution of Airbags across various car categories.
    * Car Price Distribution for Top Manufacturers.
    * Correlation Matrix Heatmap to understand relationships between numerical features.
    * Average Number of Airbags by Car Category.
    * Engine Volume Distribution by Car Category.
    * Average Price by Car Category.
* **Feature Engineering**:
    * Creation of a 'Years Since 1990' feature from 'Prod. year' to simplify time-based analysis.
    * Label Encoding for all categorical (object) columns to convert them into numerical representations.
* **Model Training and Evaluation**:
    * The target variable ('Price') is log-transformed (`np.log1p`) to handle its skewed distribution.
    * Data is split into training and testing sets (70% train, 30% test).
    * Features are scaled using `StandardScaler`.
    * Several regression models are trained and evaluated:
        * Linear Regression
        * Decision Tree Regressor
        * Random Forest Regressor
        * Gradient Boosting Regressor
        * Extra Trees Regressor
        * XGBoost Regressor
        * K-Nearest Neighbors (KNN) Regressor
        * LightGBM Regressor
    * Models are evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Cross-validation (cv=10) is performed to assess model robustness.
    * Scatter plots visualize true vs. predicted values for each model.

## Technologies and Libraries Used

* **Python**
* **pandas**: For data manipulation and analysis.
* **numpy**: For numerical operations.
* **matplotlib**: For static visualizations.
* **seaborn**: For enhanced statistical visualizations.
* **scikit-learn (sklearn)**: For preprocessing (OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler), model selection (train_test_split, cross_val_score, GridSearchCV), and various regression models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, KNeighborsRegressor).
* **xgboost**: For XGBoost Regressor.
* **lightgbm**: For LightGBM Regressor.
