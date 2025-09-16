from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd

def train_anomaly_detection_model(X_train):
    """
    Trains an anomaly detection model (One-Class SVM).
    """
    model = OneClassSVM(gamma='auto')
    model.fit(X_train)
    return model

def predict_anomalies(model, X_test):
    """
    Predicts anomalies using the trained One-Class SVM model.
    Returns -1 for anomalies and 1 for inliers.
    """
    return model.predict(X_test)

def train_failure_prediction_model(X_train, y_train):
    """
    Trains a regression model (Random Forest Regressor) for time-to-failure prediction.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_time_to_failure(model, X_test):
    """
    Predicts time-to-failure using the trained Random Forest Regressor model.
    """
    return model.predict(X_test)