# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('FINAL_SDI CSV FILE.csv')  # Replace with the actual file path
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Feature Engineering and Data Preprocessing
def preprocess_data(df):
    # Convert categorical variables to numeric using encoding (if necessary)
    df['Industry'] = df['Industry'].astype('category').cat.codes
    df['Chemical_Type'] = df['Chemical_Type'].astype('category').cat.codes
    df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes
    return df

df = preprocess_data(df)

# ARIMA Model
def arima_model(df, column):
    # Use 'Date' as the index for time series analysis
    df.set_index('Date', inplace=True)
    ts = df[column]
    
    # Fit an ARIMA model
    model = ARIMA(ts, order=(5,1,0))  # Example order, you can fine-tune it
    model_fit = model.fit()
    
    # Forecast for future years (2025-2030)
    forecast = model_fit.forecast(steps=6)  # Forecast for 6 years
    forecast_years = pd.date_range(start='2025', periods=6, freq='Y').year
    forecast_df = pd.DataFrame(forecast, index=forecast_years, columns=[column])
    
    return forecast_df

# LSTM Model for Time Series
def create_lstm_model(df, column):
    # Prepare data for LSTM
    df.set_index('Date', inplace=True)
    ts = df[column]
    ts = ts.values.reshape(-1, 1)
    ts = np.array(ts)
    
    scaler = StandardScaler()
    ts_scaled = scaler.fit_transform(ts)
    
    X = []
    y = []
    
    for i in range(60, len(ts_scaled)):
        X.append(ts_scaled[i-60:i, 0])  # Taking last 60 values for prediction
        y.append(ts_scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)
    
    # Predict future values
    future_predictions = []
    inputs = ts_scaled[len(ts_scaled) - 60:]
    inputs = inputs.reshape(1, -1)
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
    
    for _ in range(6):
        predicted_value = model.predict(inputs)
        future_predictions.append(predicted_value)
        inputs = np.append(inputs[:, 1:, :], predicted_value, axis=1)
    
    # Reverse scaling to get the original values
    future_predictions = scaler.inverse_transform(future_predictions)
    future_df = pd.DataFrame(future_predictions, index=pd.date_range(start='2025', periods=6, freq='Y').year, columns=[column])
    
    return future_df

# Random Forest Regressor Model
def regressor_model(df):
    X = df.drop(columns=['Date', 'Effluent_Volume_Liters'])  # Features
    y = df['Effluent_Volume_Liters']  # Target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae}")
    
    return model

# Streamlit UI and Model Deployment
def app():
    st.title("Effluent Prediction Model")
    
    # Input from user for industry-specific parameters
    industry = st.selectbox("Select Industry", df['Industry'].unique())
    location = st.text_input("Enter Location", "Delhi")
    pH = st.number_input("Enter pH Value", min_value=0.0, max_value=14.0, value=7.0)
    tds = st.number_input("Enter TDS (ppm)", min_value=0, max_value=2000, value=500)
    conductivity = st.number_input("Enter Conductivity (µS)", min_value=0, max_value=5000, value=1000)
    
    # Run the models and show results
    if st.button("Predict Future Trends"):
        future_arima = arima_model(df, 'Effluent_Volume_Liters')
        future_lstm = create_lstm_model(df, 'Effluent_Volume_Liters')
        
        st.write("ARIMA Prediction for Effluent Concentration (2025-2030):")
        st.write(future_arima)
        
        st.write("LSTM Prediction for Effluent Concentration (2025-2030):")
        st.write(future_lstm)
        
        st.write("Random Forest Regressor Prediction:")
        rf_model = regressor_model(df)
        industry_data = df[df['Industry'] == industry].drop(columns=['Date'])
        prediction = rf_model.predict(industry_data)
        st.write(f"Predicted Effluent Volume for selected Industry and Parameters: {prediction}")
    
    # Display data visualization
    st.subheader("Data Visualization")
    st.line_chart(df[['Date', 'Effluent_Volume_Liters']].set_index('Date'))

# Run the Streamlit app
if __name__ == "__main__":
    app()

print("worked successfully")