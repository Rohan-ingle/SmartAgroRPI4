import sqlite3
import os
from datetime import datetime, date
import numpy as np
import joblib
import sys
import argparse
import time
import requests # Added import for live API call

# --- Configuration ---
# Reusing relevant configurations from simulation
WEATHER_DB_PATH = 'weather_data.db'
MODEL_PATH = 'weather_xgboost_best_model.joblib'
SCALER_PATH = 'weather_xgboost_scaler.joblib'
WATER_THRESHOLD_PRECIP = 0.5
SOIL_DRY_THRESHOLD = 40
WEATHER_FEATURE_COLUMNS = ['temp', 'humidity', 'pressure', 'wind_speed']
# Add API configuration needed for live forecast
# Replace with your actual API key and location
LIVE_API_KEY = "YOUR_WEATHERAPI_KEY" # IMPORTANT: Replace with your key
LIVE_API_LAT = 40.7128 # Example: New York Latitude
LIVE_API_LON = -74.0060 # Example: New York Longitude
LIVE_API_FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"
# --- End Configuration ---

# --- Sensor Reading (Placeholder) ---
def get_current_soil_moisture():
    """Placeholder function to simulate reading soil moisture sensor."""
    # In a real scenario, this would interact with hardware.
    # Simulate fluctuating moisture for testing
    simulated_moisture = np.random.uniform(30, 60) # Random value between 30% and 60%
    print(f"(Placeholder) Current Soil Moisture: {simulated_moisture:.2f}%")
    return simulated_moisture

# --- Get Weather Forecast for Prediction ---
def get_weather_forecast_for_prediction(db_path):
    """Fetches the most recent forecast data matching the model features from the DB."""
    if not os.path.exists(db_path):
        print(f"Error: Weather database {db_path} not found.")
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        query = f"SELECT {', '.join(WEATHER_FEATURE_COLUMNS)} FROM weather_forecast ORDER BY date DESC LIMIT 1"
        cursor.execute(query)
        row = cursor.fetchone()
        if row:
            processed_row = [0.0 if v is None else v for v in row] # Replace None with 0.0
            return np.array([processed_row], dtype=np.float32)  # Return as a 2D array
        else:
            print("No forecast data found in weather database.")
            return None
    except sqlite3.OperationalError as e:
        print(f"Error querying weather database (check table/columns): {e}")
        return None
    except Exception as e:
        print(f"Error fetching forecast data from DB: {e}")
        return None
    finally:
        conn.close()

# --- Get Live Weather Forecast from API ---
def get_live_api_forecast(api_key, lat, lon, forecast_url):
    """Fetches live precipitation forecast for today from WeatherAPI."""
    params = {
        'key': api_key,
        'q': f"{lat},{lon}",
        'days': 1 # Forecast for today
    }
    try:
        response = requests.get(forecast_url, params=params, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'forecast' in data and 'forecastday' in data['forecast'] and len(data['forecast']['forecastday']) > 0:
            today_forecast = data['forecast']['forecastday'][0]
            if 'day' in today_forecast and 'totalprecip_mm' in today_forecast['day']:
                precip_mm = float(today_forecast['day']['totalprecip_mm'])
                print(f"Live API Forecast: Precipitation today = {precip_mm:.2f} mm")
                return precip_mm
            else:
                print("API Response structure unexpected (missing day/totalprecip_mm).")
                return None
        else:
            print("API Response structure unexpected (missing forecast/forecastday).")
            return None
    except requests.exceptions.Timeout:
        print("Error fetching live API forecast: Request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live API forecast: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error parsing live API forecast data: {e}")
        return None

# --- Get Actual Weather for Feedback (Optional for single run, kept for context) ---
def get_actual_weather_for_date(db_path, target_date):
    """Fetches the actual (or most accurate available) weather data for a specific date."""
    if not os.path.exists(db_path):
        print(f"Error: Weather database {db_path} not found for feedback.")
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    date_str = target_date.strftime('%Y-%m-%d')
    try:
        query = f"SELECT totalprecip_mm FROM weather_forecast WHERE date = ? LIMIT 1"
        cursor.execute(query, (date_str,))
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
        else:
            return None
    except sqlite3.OperationalError as e:
        print(f"Error querying weather database for actuals (check table/columns): {e}")
        return None
    except Exception as e:
        print(f"Error fetching actual weather data: {e}")
        return None
    finally:
        conn.close()

# --- Predict Precipitation using Trained Model ---
def predict_precipitation_from_db_forecast(db_forecast_input):
    """Predicts precipitation using the trained model and DB forecast input."""
    if db_forecast_input is None:
        return None

    if np.isnan(db_forecast_input).any():
        print("Warning: NaN values found in forecast input from DB, filling with 0.")
        db_forecast_input = np.nan_to_num(db_forecast_input, nan=0.0)

    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print(f"Error: Model ({MODEL_PATH}) or Scaler ({SCALER_PATH}) not found.")
            return None
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None

    predicted_precip = None
    try:
        X_scaled = scaler.transform(db_forecast_input)
        predicted_precip = model.predict(X_scaled)[0]
        predicted_precip = max(0, predicted_precip) # Ensure non-negative prediction
        print(f"Model Prediction (from DB forecast): Precipitation today = {predicted_precip:.2f} mm")
        return predicted_precip
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

# --- Make Watering Decision ---
def make_watering_decision(current_soil_moisture, predicted_precip):
    """
    Makes a watering decision based on current soil moisture and predicted precipitation.

    Args:
        current_soil_moisture (float): The current soil moisture percentage.
        predicted_precip (float): The predicted precipitation in mm for today.

    Returns:
        bool: True if watering is needed, False otherwise.
    """
    if current_soil_moisture is None or predicted_precip is None:
        print("Cannot make decision: Missing soil moisture or precipitation prediction.")
        return False

    should_water = False
    print(f"Making decision: Soil Moisture={current_soil_moisture:.2f}%, Predicted Precip={predicted_precip:.2f}mm")
    if predicted_precip < WATER_THRESHOLD_PRECIP and current_soil_moisture < SOIL_DRY_THRESHOLD:
        print("Decision: Water the plant (Low predicted rain, dry soil).")
        should_water = True
    elif predicted_precip >= WATER_THRESHOLD_PRECIP:
        print("Decision: Do not water the plant (Sufficient rain predicted).")
        should_water = False
    else:
        print("Decision: Do not water the plant (Low predicted rain, but soil is moist).")
        should_water = False

    return should_water
