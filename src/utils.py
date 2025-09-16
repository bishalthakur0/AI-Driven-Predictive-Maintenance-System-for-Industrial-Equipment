import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by cleaning, normalizing, and detecting outliers.
    """
    # 1. Data Cleaning (Example: handling missing values)
    df = df.dropna()  # Drop rows with any missing values

    # 2. Data Normalization (Example: using StandardScaler)
    # Assuming numerical features are to be scaled
    numerical_cols = df.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 3. Outlier Detection (Example: using Isolation Forest)
    # This is a basic example; thresholding and handling outliers would be more sophisticated in a real system
    if not numerical_cols.empty:
        iso_forest = IsolationForest(random_state=42)
        outlier_predictions = iso_forest.fit_predict(df[numerical_cols])
        df['is_outlier'] = outlier_predictions == -1  # -1 for outliers, 1 for inliers

    return df