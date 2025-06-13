# Car Price Prediction Streamlit App

![Project Status: Completed](https://img.shields.io/badge/Status-Completed-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
- [Performance Metrics](#performance-metrics)
- [How It Works](#how-it-works)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project is a machine learning web application built with Streamlit that predicts the selling price of used cars. It leverages a robust Random Forest Regressor model trained on a comprehensive dataset of car listings. The application aims to provide a user-friendly interface for estimating car prices based on key features such as car name, year of manufacture, kilometers driven, fuel type, seller type, transmission, and owner history.

This project covers the end-to-end machine learning lifecycle, from data cleaning and preprocessing to model training, evaluation, and deployment.

## Features

- **Interactive UI:** User-friendly interface built with Streamlit for easy input of car details.
- **Real-time Predictions:** Get instant price predictions based on the trained ML model.
- **Comprehensive Preprocessing:** Handles feature engineering (e.g., 'Car_Age', 'brand' extraction), ordinal encoding ('owner'), and one-hot encoding for categorical features, all integrated into a scikit-learn pipeline for consistent data transformation.
- **Robust Model:** Utilizes a Random Forest Regressor, known for its strong performance in regression tasks.

## Dataset

The model was trained on the `car_dekho_data.csv` dataset, which contains various attributes of used cars.
Key features used for prediction include:
- `Car_Name`
- `Year`
- `selling_price` (Target variable)
- `km_driven`
- `fuel`
- `seller_type`
- `transmission`
- `owner`

### Data Preprocessing Highlights:
- **Feature Engineering:** 'Car_Age' calculated from 'Year'; 'brand' extracted from 'Car_Name'.
- **Categorical Encoding:** 'owner' handled with ordinal mapping; 'brand', 'fuel', 'seller_type', 'transmission' handled with One-Hot Encoding.
- **Target Transformation:** 'selling_price' was Log-transformed (`np.log1p`) to handle skewness and improve model performance.
- **Feature Scaling:** Numerical features were scaled using `StandardScaler`.

## Machine Learning Model

The core of this predictor is a **Random Forest Regressor** model. It was chosen for its high predictive accuracy and ability to handle non-linear relationships within the data. The entire preprocessing pipeline, including feature scaling and the Random Forest model, is encapsulated within a `scikit-learn Pipeline` for streamlined deployment.

## Performance Metrics

The final model's performance was evaluated on a dedicated test set (original price scale):

- **Final R2 Score on Test Set:** `0.7193`
- **Final MAE on Test Set:** `122213.00 INR`

These metrics indicate that the model explains approximately 71.93% of the variance in car selling prices and, on average, predicts prices within `122,213 INR` of the actual selling price. The model demonstrates good generalization capabilities and effectively avoids significant overfitting.

## How It Works

1.  **User Input:** The Streamlit app collects car details from the user via an interactive form.
2.  **Data Preprocessing:** The input data undergoes the same preprocessing steps as the training data, including feature engineering, encoding, and scaling, handled seamlessly by the loaded `scikit-learn Pipeline`.
3.  **Prediction:** The preprocessed data is fed into the trained Random Forest Regressor model, which outputs a predicted selling price (in Log scale).
4.  **Inverse Transformation:** The predicted log price is then inverse-transformed back to the original Indian Rupee (INR) scale using `np.expm1()` before being displayed to the user.

## Installation and Setup

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/palmyz000/Car_Dekho_Price_Regression.git](https://github.com/palmyz000/Car_Dekho_Price_Regression.git)
    cd Car_Dekho_Price_Regression
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure model files are present:**
    The `car_price_prediction_pipeline.joblib` and `feature_names.joblib` files (generated during model training and saved for deployment) must be in the project root directory.

## Usage

After completing the installation and setup, you can run the Streamlit application:

```bash
streamlit run app.py
