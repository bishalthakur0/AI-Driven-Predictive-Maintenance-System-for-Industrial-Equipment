import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from src.utils import preprocess_data
from src.model import train_anomaly_detection_model, predict_anomalies, train_failure_prediction_model, predict_time_to_failure

def main():
    st.title("AI-Driven Predictive Maintenance System")
    st.write("Welcome to the predictive maintenance dashboard.")

    # Placeholder for data ingestion
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload sensor data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully:")
        st.dataframe(df.head())
    else:
        st.info("No file uploaded. Using sample_sensor_data.csv for demonstration.")
        try:
            df = pd.read_csv("data/sample_sensor_data.csv")
            st.write("Sample data loaded successfully:")
            st.dataframe(df.head())
        except FileNotFoundError:
            st.error("sample_sensor_data.csv not found. Please upload a CSV file or ensure the sample data exists.")
            return

        # Data Preprocessing
        st.header("2. Data Preprocessing")
        st.write("Performing data cleaning, normalization, and outlier detection...")
        processed_df = preprocess_data(df.copy())
        st.write("Data processed successfully. First 5 rows of processed data:")
        st.dataframe(processed_df.head())
        st.write("Outliers detected:", processed_df['is_outlier'].sum())

        # ML Prediction Engine
        st.header("3. ML Prediction Engine")
        st.write("Training anomaly detection and time-to-failure prediction models...")

        # Assuming the processed_df has features for ML models and a 'failure_time' target for regression
        # For demonstration, let's create dummy 'failure_time' and features if they don't exist
        if 'feature_1' not in processed_df.columns:
            st.warning("Creating dummy features and 'failure_time' for demonstration. Please ensure your CSV has relevant columns.")
            processed_df['feature_1'] = np.random.rand(len(processed_df))
            processed_df['feature_2'] = np.random.rand(len(processed_df))
            processed_df['failure_time'] = np.random.randint(1, 100, len(processed_df))

        features = ['feature_1', 'feature_2'] # Replace with actual feature columns from your data
        target = 'failure_time'

        if all(col in processed_df.columns for col in features + [target]):
            X = processed_df[features]
            y = processed_df[target]

            # Anomaly Detection
            st.subheader("Anomaly Detection")
            anomaly_model = train_anomaly_detection_model(X)
            anomalies = predict_anomalies(anomaly_model, X)
            processed_df['anomaly'] = anomalies
            st.write(f"Number of anomalies detected: {(processed_df['anomaly'] == -1).sum()}")
            st.dataframe(processed_df[processed_df['anomaly'] == -1].head())

            # Failure Prediction
            st.subheader("Failure Prediction")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            failure_model = train_failure_prediction_model(X_train, y_train)
            predictions = predict_time_to_failure(failure_model, X_test)
            st.write("Sample failure predictions:", predictions[:5])
            st.write("Model trained and predictions made.")
        else:
            st.error("Missing required feature or target columns for ML models. Please check your data.")

        # Dashboard & Visualization
        st.header("4. Dashboard & Visualization")
        st.write("Displaying equipment health status, failure prediction timeline, and trend analysis.")

        if 'anomaly' in processed_df.columns:
            st.subheader("Equipment Health Status")
            num_anomalies = (processed_df['anomaly'] == -1).sum()
            total_readings = len(processed_df)
            if total_readings > 0:
                anomaly_percentage = (num_anomalies / total_readings) * 100
                st.metric(label="Anomaly Rate", value=f"{anomaly_percentage:.2f}%")
                if anomaly_percentage > 10: # Example threshold
                    st.error("Equipment Health: Critical - High Anomaly Rate")
                elif anomaly_percentage > 2:
                    st.warning("Equipment Health: Warning - Moderate Anomaly Rate")
                else:
                    st.success("Equipment Health: Normal - Low Anomaly Rate")
            else:
                st.info("No data to assess equipment health.")

            st.subheader("Anomaly Detection Visualization")
            st.line_chart(processed_df['anomaly'])

        if 'failure_time' in processed_df.columns and 'predictions' in locals():
            st.subheader("Failure Prediction Timeline")
            # For simplicity, let's just show a histogram of predicted failure times
            st.bar_chart(pd.Series(predictions).value_counts().sort_index())
            st.write("This chart shows the distribution of predicted days to failure.")

        # Alerts & Notifications
        st.header("5. Alerts & Notifications")
        st.write("Generating real-time alerts based on anomaly detection...")

        if 'anomaly' in processed_df.columns:
            critical_anomalies = processed_df[processed_df['anomaly'] == -1]
            if not critical_anomalies.empty:
                st.error(f"ALERT: {len(critical_anomalies)} anomalies detected! Immediate attention required.")
                st.dataframe(critical_anomalies.head())
            else:
                st.success("No critical anomalies detected. Equipment operating normally.")

        # Placeholder for additional reporting features
        st.write("Reports for managers and engineers can be generated here.")

if __name__ == "__main__":
    main()