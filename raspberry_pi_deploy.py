import os
import io
import time
import threading
import markdown # Import the markdown library
# Use session to store results across requests
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session, send_from_directory, flash
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import base64
import random
import sqlite3
import subprocess
from datetime import datetime, date
from dotenv import load_dotenv  # Add this import

# --- Load environment variables from .env file ---
load_dotenv()  # This will load variables from .env into os.environ

WEATHER_DB_PATH = "data/weather_data.db"         # For local DHT11 data
WEATHERAPI_DB_PATH = "data/weatherapi_data.db"   # For WeatherAPI.com data
WEATHER_TABLE = "weather_forecast"
WEATHERAPI_TABLE = "weatherapi_forecast"

AUTO_TRAINING = False
AUTO_WATERING_KEY = "auto_watering_enabled"

UPLOAD_FOLDER = "static/uploads"
ANNOTATED_FOLDER = "static/annotated"
# --- Flask App & SocketIO ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='threading')

def predict_and_water():
    """Predict if watering is needed and water if required."""
    soil_wet = read_soil_moisture()
    # If soil is wet, do not water
    if soil_wet:
        print("Soil is wet. No need to water.")
        return False
    # Predict precipitation using the trained model
    try:
        import joblib
        MODEL_PATH = "weather_xgboost_best_model.joblib"
        SCALER_PATH = "weather_xgboost_scaler.joblib"
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print("Weather model or scaler not found. Skipping prediction.")
            return False
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        # Fetch latest weather features from DB
        conn = sqlite3.connect(WEATHER_DB_PATH)
        cursor = conn.cursor()
        # Adjust columns as per your model's features
        feature_cols = ['temp', 'humidity']
        cursor.execute(f"SELECT {', '.join(feature_cols)} FROM {WEATHER_TABLE} ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if not row:
            print("No weather data available for prediction.")
            return False
        X = np.array([row], dtype=np.float32)
        X_scaled = scaler.transform(X)
        predicted_precip = model.predict(X_scaled)[0]
        print(f"Predicted precipitation: {predicted_precip:.2f} mm")
        # If predicted precipitation is low, water the plants
        if predicted_precip < 1.0:  # Threshold, adjust as needed
            print("Need to water. Watering plants (activating buzzer).")
            threading.Thread(target=activate_buzzer, args=(BUZZER_SPRINKLER_PIN,)).start()
            return True
        else:
            print("No need to water based on prediction.")
            return False
    except Exception as e:
        print(f"Error in prediction/watering: {e}")
        return False

def auto_watering_worker():
    """Background thread: run prediction and watering twice daily if enabled."""
    last_run = None
    while True:
        now = datetime.now()
        # --- FIX: Use a global variable for auto-watering state, not session ---
        enabled = False
        try:
            with app.app_context():
                enabled = app.config.get("AUTO_WATERING_ENABLED", False)
        except Exception:
            enabled = False
        # If enabled, run at 6am and 6pm
        if enabled:
            if last_run != now.date() or (now.hour in [6, 18] and (last_run is None or last_run != (now.date(), now.hour))):
                print(f"Auto-watering check at {now}")
                predict_and_water()
                last_run = (now.date(), now.hour)
        time.sleep(60 * 30)  # Check every 30 minutes

@app.route("/toggle_auto_watering", methods=["POST"])
def toggle_auto_watering():
    # --- FIX: Store state in app.config for background thread access ---
    enabled = app.config.get("AUTO_WATERING_ENABLED", False)
    app.config["AUTO_WATERING_ENABLED"] = not enabled
    msg = f"Automated watering {'enabled' if not enabled else 'disabled'}."
    status_text = "(Enabled)" if not enabled else "(Disabled)"
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "enabled": not enabled,
            "status_text": status_text,
            "message": msg
        })
    flash(msg)
    return redirect(url_for('index'))

@app.route("/manual_water", methods=["POST"])
def manual_water():
    watered = predict_and_water()
    msg = "Manual watering triggered." if watered else "No need to water (soil wet or rain predicted)."
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"message": msg})
    flash(msg)
    return redirect(url_for('index'))

def ensure_weather_table(db_path, table, columns):
    """Ensure the given table exists in the specified database with required columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if not existing_cols:
        # Table does not exist, create it
        col_defs = ", ".join([f"{col} {typ}" for col, typ in columns])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})")
    else:
        # Table exists, add missing columns
        for col, typ in columns:
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                    print(f"Added missing column '{col}' to {table}")
                except sqlite3.OperationalError as e:
                    print(f"Could not add column {col}: {e}")
    conn.commit()
    conn.close()

def insert_dht_reading(temp, humidity):
    """Insert a DHT reading into the local weather database."""
    columns = [
        ("timestamp", "TEXT"),
        ("date", "TEXT"),
        ("temp", "REAL"),
        ("humidity", "REAL")
    ]
    ensure_weather_table(WEATHER_DB_PATH, WEATHER_TABLE, columns)
    conn = sqlite3.connect(WEATHER_DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute(
        f"INSERT INTO {WEATHER_TABLE} (timestamp, date, temp, humidity) VALUES (?, ?, ?, ?)",
        (now.isoformat(), now.date().isoformat(), temp, humidity)
    )
    conn.commit()
    conn.close()

def insert_weatherapi_data(api_json):
    """Insert WeatherAPI.com data into its own database."""
    columns = [
        ("timestamp", "TEXT"),
        ("date", "TEXT"),
        ("location", "TEXT"),
        ("temp_c", "REAL"),
        ("humidity", "REAL"),
        ("condition", "TEXT"),
        ("precip_mm", "REAL")
    ]
    ensure_weather_table(WEATHERAPI_DB_PATH, WEATHERAPI_TABLE, columns)
    conn = sqlite3.connect(WEATHERAPI_DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()
    try:
        location = api_json['location']['name'] + ", " + api_json['location']['country']
        temp_c = api_json['current']['temp_c']
        humidity = api_json['current']['humidity']
        condition = api_json['current']['condition']['text']
        precip_mm = api_json['current'].get('precip_mm', 0.0)
        date_str = api_json['location']['localtime'].split()[0] if 'localtime' in api_json['location'] else now.date().isoformat()
        cursor.execute(
            f"INSERT INTO {WEATHERAPI_TABLE} (timestamp, date, location, temp_c, humidity, condition, precip_mm) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now.isoformat(), date_str, location, temp_c, humidity, condition, precip_mm)
        )
        conn.commit()
    except Exception as e:
        print(f"Error inserting WeatherAPI data: {e}")
    finally:
        conn.close()

def retrain_weather_model_daily():
    """Background thread: retrain the weather model once per day."""
    last_trained = None
    while True:
        today = date.today()
        if last_trained != today:
            print(f"Retraining weather model for {today}...")
            try:
                subprocess.run(
                    ["python3", "weather_train.py", "--db", WEATHER_DB_PATH],
                    cwd="/home/rpi/SmartAgro",
                    check=True
                )
                print("Weather model retrained.")
            except Exception as e:
                print(f"Error retraining weather model: {e}")
            last_trained = today
        time.sleep(3600)  # Check every hour

# --- Hardware (GPIO) ---
import RPi.GPIO as GPIO
import board
import adafruit_dht

# --- ML/AI ---
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import requests

# --- Configuration ---
DHT_PIN = board.D4
SOIL_PIN = 17   # BCM 17 (physical pin 11) for soil sensor
BUZZER_SPRINKLER_PIN = 27
BUZZER_ANIMAL_PIN = 22
ANIMAL_LED_PIN = 23  # New: GPIO pin for animal intrusion LED
YOLO_MODEL_PATH = 'models/wildlife_detector.pt'
PLANT_MODEL_PATH = 'models/plant_disease_efficientnetb0_best.pth'
GEMINI_API_KEY = os.environ.get('GEMINI_API')
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'
WEATHER_API_KEY = os.environ.get('WEATHERAPI_KEY')
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'
WEATHER_LOCATION = 'Pune'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_SPRINKLER_PIN, GPIO.OUT)
GPIO.setup(BUZZER_ANIMAL_PIN, GPIO.OUT)
GPIO.setup(SOIL_PIN, GPIO.IN)
GPIO.setup(ANIMAL_LED_PIN, GPIO.OUT)  # New: Setup LED pin

# --- Initialize DHT Sensor ---
try:
    dht_device = adafruit_dht.DHT11(DHT_PIN)
    print("DHT11 Sensor Initialized using adafruit-circuitpython-dht.")
except Exception as e:
    print(f"Error: Failed to initialize DHT11 sensor with adafruit-circuitpython-dht: {e}")
    print("Sensor readings will be unavailable.")
    dht_device = None

# --- Load Models ---
from ultralytics import YOLO

yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.eval()

plant_checkpoint = torch.load(PLANT_MODEL_PATH, map_location='cpu')
plant_class_names = plant_checkpoint['class_names']
num_classes = len(plant_class_names)
plant_model = efficientnet_b0(weights=None)
num_ftrs = plant_model.classifier[1].in_features
import torch.nn as nn
plant_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_ftrs, num_classes)
)
plant_model.load_state_dict(plant_checkpoint['model_state_dict'])
plant_model.eval()

plant_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Animal Intrusion State ---
animal_intrusion_state = {
    "detected": False,
    "last_detection_time": 0,
    "confidence_sum": 0.0,
    "detection_count": 0,
    "notification_sent": False
}
ANIMAL_INTRUSION_CONFIDENCE_THRESHOLD = 0.5
ANIMAL_INTRUSION_ACCUM_THRESHOLD = 2.0
ANIMAL_INTRUSION_MIN_COUNT = 3
ANIMAL_INTRUSION_RESET_SECONDS = 60

# --- Helper Functions ---
def read_dht11():
    global dht_device
    # Try DHT11 sensor first
    try:
        if dht_device is not None:
            temperature_c = dht_device.temperature
            humidity      = dht_device.humidity
            if temperature_c is not None and humidity is not None:
                insert_dht_reading(temperature_c, humidity)
                return temperature_c, humidity
    except Exception as e:
        print(f"DHT11 error: {e}")

    # If DHT11 fails, try WeatherAPI
    print("DHT11 unavailable or failed, trying WeatherAPI...")
    weather_data = get_weatherapi_data()
    if weather_data:
        try:
            temp = weather_data['current']['temp_c']
            hum = weather_data['current']['humidity']
            print(f"Using WeatherAPI values: T={temp}, H={hum}")
            return temp, hum
        except Exception as e:
            print(f"Error extracting WeatherAPI values: {e}")

    # If both fail, simulate values
    temp_sim = round(random.uniform(20.0, 30.0), 1)
    hum_sim  = round(random.uniform(40.0, 60.0), 1)
    print(f"Simulating sensor values: T={temp_sim}, H={hum_sim}")
    insert_dht_reading(temp_sim, hum_sim)
    return temp_sim, hum_sim

def read_soil_moisture():
    return GPIO.input(SOIL_PIN) == GPIO.LOW

def activate_buzzer(pin, duration=2):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)

def activate_buzzer_and_led(buzzer_pin, led_pin, duration=2):
    """Activate both buzzer and LED for animal intrusion."""
    GPIO.output(buzzer_pin, GPIO.HIGH)
    GPIO.output(led_pin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(buzzer_pin, GPIO.LOW)
    GPIO.output(led_pin, GPIO.LOW)

def check_sprinkler_and_buzzer():
    if not read_soil_moisture():
        threading.Thread(target=activate_buzzer, args=(BUZZER_SPRINKLER_PIN,)).start()
        return True
    return False

def detect_animal_on_frame(frame):
    results = yolo_model(frame)
    detected_in_frame = False
    max_conf_in_frame = 0.0
    if results and len(results) > 0:
        res = results[0]
        boxes = res.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            if conf > max_conf_in_frame:
                max_conf_in_frame = conf
            if conf >= ANIMAL_INTRUSION_CONFIDENCE_THRESHOLD:
                detected_in_frame = True
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    now = time.time()
    if now - animal_intrusion_state["last_detection_time"] > ANIMAL_INTRUSION_RESET_SECONDS:
        animal_intrusion_state["confidence_sum"] = 0.0
        animal_intrusion_state["detection_count"] = 0
        animal_intrusion_state["notification_sent"] = False

    if detected_in_frame:
        animal_intrusion_state["confidence_sum"] += max_conf_in_frame
        animal_intrusion_state["detection_count"] += 1
        animal_intrusion_state["last_detection_time"] = now
        if (not animal_intrusion_state["notification_sent"] and
            animal_intrusion_state["confidence_sum"] >= ANIMAL_INTRUSION_ACCUM_THRESHOLD and
            animal_intrusion_state["detection_count"] >= ANIMAL_INTRUSION_MIN_COUNT):
            animal_intrusion_state["notification_sent"] = True
            # Activate both buzzer and LED for animal intrusion
            socketio.start_background_task(
                target=activate_buzzer_and_led,
                buzzer_pin=BUZZER_ANIMAL_PIN,
                led_pin=ANIMAL_LED_PIN
            )
            animal_intrusion_state["detected"] = True

    return frame, detected_in_frame, max_conf_in_frame

def detect_animal_and_annotate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False, os.path.basename(image_path), 0.0

    annotated_img, detected, max_conf = detect_animal_on_frame(img.copy())

    annotated_filename = secure_filename(os.path.basename(image_path))
    annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_filename)
    try:
        cv2.imwrite(annotated_path, annotated_img)
    except Exception as e:
        print(f"Error saving annotated image to {annotated_path}: {e}")
        annotated_filename = os.path.basename(image_path)

    return detected, annotated_filename, max_conf

def classify_plant_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = plant_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = plant_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return plant_class_names[pred.item()], conf.item()

def gemini_qna(disease_name):
    # --- Debug: Check if API Key is loaded ---
    api_key = GEMINI_API_KEY
    if not api_key:
        print("Error: GEMINI_API environment variable not set.")
        return "Error: Gemini API key is missing."
    # print(f"Using Gemini API Key: {api_key[:4]}...{api_key[-4:]}") # Optional: Print partial key

    prompt = f"Give a brief overview and mitigation steps for the plant disease: {disease_name}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {"key": api_key} # Use the loaded key

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=15) # Add timeout

        if response.ok:
            try:
                # --- Debug: Print successful response structure (optional) ---
                # print("Gemini Response JSON:", response.json())
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                print(f"Error parsing Gemini response: {e}")
                print("Gemini Raw Response:", response.text)
                return "Error parsing answer from Gemini."
        else:
            # --- Debug: Print failure details ---
            print(f"Error contacting Gemini API. Status Code: {response.status_code}")
            print("Gemini Error Response:", response.text)
            return f"Failed to contact Gemini API (Status: {response.status_code}). Check logs for details."

    except requests.exceptions.RequestException as e:
        # --- Debug: Print network or request errors ---
        print(f"Error during Gemini API request: {e}")
        return f"Network error contacting Gemini API: {e}"
    except Exception as e:
        # --- Debug: Catch any other unexpected errors ---
        print(f"Unexpected error in gemini_qna: {e}")
        return "An unexpected error occurred while contacting Gemini."

def get_weatherapi_data():
    """Fetch weather data for Pune from WeatherAPI.com and store it."""
    if not WEATHER_API_KEY:
        print("WeatherAPI.com API key not set in environment.")
        return None
    params = {
        'key': WEATHER_API_KEY,
        'q': WEATHER_LOCATION,
        'aqi': 'no'
    }
    try:
        resp = requests.get(WEATHER_API_URL, params=params, timeout=10)
        if resp.ok:
            api_json = resp.json()
            # Defensive: check for expected keys
            if 'current' in api_json and 'location' in api_json:
                insert_weatherapi_data(api_json)  # Store in weatherapi_data.db
                return api_json
            else:
                print(f"WeatherAPI.com response missing expected keys: {api_json}")
                return None
        else:
            print(f"WeatherAPI.com error: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        print(f"WeatherAPI.com exception: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    temperature, humidity = read_dht11()
    soil_wet = read_soil_moisture()
    animal_intrusion = animal_intrusion_state.get("notification_sent", False)
    temp_display = f"{temperature:.1f}" if temperature is not None else "N/A"
    humidity_display = f"{humidity:.1f}" if humidity is not None else "N/A"

    weather_data = None
    # Only fetch WeatherAPI for richer display if DHT11 and weatherapi fallback both failed
    if (temperature == "N/A" or humidity == "N/A"):
        weather_data = get_weatherapi_data()

    animal_annotated = session.pop('animal_annotated', None)
    animal_confidence = session.pop('animal_confidence', None)
    animal_image_url = url_for('static', filename='annotated/' + animal_annotated) if animal_annotated else None
    plant_result = session.pop('plant_result', None)
    gemini_answer_html = session.pop('gemini_answer_html', None)
    cctv_active = cctv_capture["active"]
    cctv_image_url = url_for('cctv_latest') if cctv_capture.get("latest_annotated") else None

    # --- FIX: Use app.config for auto_watering_enabled ---
    auto_watering_enabled = app.config.get("AUTO_WATERING_ENABLED", False)

    return render_template(
        'index.html',
        temperature=temp_display,
        humidity=humidity_display,
        soil_wet=soil_wet,
        animal_annotated=animal_annotated,
        animal_confidence=animal_confidence,
        animal_image_url=animal_image_url,
        animal_intrusion=animal_intrusion,
        plant_result=plant_result,
        gemini_answer_html=gemini_answer_html,
        cctv_active=cctv_active,
        cctv_image_url=cctv_image_url,
        weather_data=weather_data,
        WEATHER_API_KEY=WEATHER_API_KEY,
        auto_watering_enabled=auto_watering_enabled
    )

@app.route("/sprinkler", methods=["POST"])
def sprinkler():
    triggered = check_sprinkler_and_buzzer()
    return redirect(url_for('index'))

@app.route("/animal", methods=["POST"])
def animal():
    file = request.files.get('animal_image')
    if not file or not file.filename:
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(image_path)
    except Exception as e:
        print(f"Error saving uploaded file {filename}: {e}")
        return redirect(url_for('index'))

    detected, annotated_file, max_conf = detect_animal_and_annotate(image_path)

    # Store results in session
    session['animal_annotated'] = annotated_file
    session['animal_confidence'] = max_conf

    return redirect(url_for('index')) # Redirect instead of rendering

@app.route("/plant", methods=["POST"])
def plant():
    file = request.files['plant_image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    result = classify_plant_disease(image_path)

    # Store result in session
    session['plant_result'] = result
    # Clear previous QnA answer if a new plant is classified
    session.pop('gemini_answer', None)

    return redirect(url_for('index')) # Redirect instead of rendering

@app.route("/qna", methods=["POST"])
def qna():
    disease_name = request.form['disease_name']
    answer_markdown = gemini_qna(disease_name)

    # Convert markdown to HTML
    answer_html = markdown.markdown(answer_markdown, extensions=['fenced_code', 'tables']) if answer_markdown else None

    # Store HTML results in session
    session['gemini_answer_html'] = answer_html
    # Keep the disease name for display
    session['plant_result'] = (disease_name, 1.0) # Assuming 1.0 confidence for display consistency

    return redirect(url_for('index')) # Redirect instead of rendering

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

# --- CCTV Capture State ---
cctv_capture = {
    "active": False,
    "thread": None,
    "latest_annotated": None
}

# --- Webcam Capture State ---
webcam_capture = {
    "active": False,
    "thread": None,
    "latest_annotated": None
}

def cctv_capture_worker():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        cctv_capture["active"] = False
        return
    try:
        while cctv_capture["active"]:
            ret, frame = cap.read()
            if not ret:
                print("Warning: empty frame grabbed")
                time.sleep(2)
                continue
            annotated_frame, _, _ = detect_animal_on_frame(frame)
            annotated_path = os.path.join(ANNOTATED_FOLDER, "cctv_latest.jpg")
            cv2.imwrite(annotated_path, annotated_frame)
            cctv_capture["latest_annotated"] = "cctv_latest.jpg"
            time.sleep(2.0)
    finally:
        cap.release()

@app.route("/cctv/start", methods=["POST"])
def cctv_start():
    if not cctv_capture["active"]:
        cctv_capture["active"] = True
        cctv_capture["thread"] = threading.Thread(
            target=cctv_capture_worker, daemon=True
        )
        cctv_capture["thread"].start()
    return redirect(url_for('index'))

@app.route("/cctv/stop", methods=["POST"])
def cctv_stop():
    cctv_capture["active"] = False
    cctv_capture["thread"] = None
    return redirect(url_for('index'))

@app.route("/cctv/latest")
def cctv_latest():
    fname = cctv_capture.get("latest_annotated")
    if fname:
        return send_from_directory(ANNOTATED_FOLDER, fname)
    return '', 404

@app.route("/webcam/latest")
def webcam_latest():
    fname = webcam_capture.get("latest_annotated")
    if fname:
        return send_from_directory(ANNOTATED_FOLDER, fname)
    return '', 404

# Add these to resolve url_for errors:
@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    if not webcam_capture["active"]:
        webcam_capture["active"] = True
        webcam_capture["thread"] = threading.Thread(
            target=webcam_capture_worker, daemon=True
        )
        webcam_capture["thread"].start()
    return redirect(url_for('index'))

@app.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    webcam_capture["active"] = False
    webcam_capture["thread"] = None
    return redirect(url_for('index'))

# USB webcam capture worker (device index 1)
def webcam_capture_worker():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open USB webcam")
        webcam_capture["active"] = False
        return
    try:
        while webcam_capture["active"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            annotated_frame, _, _ = detect_animal_on_frame(frame)
            path = os.path.join(ANNOTATED_FOLDER, "webcam_latest.jpg")
            cv2.imwrite(path, annotated_frame)
            webcam_capture["latest_annotated"] = "webcam_latest.jpg"
            time.sleep(2.0)
    finally:
        cap.release()

# --- Start the daily retraining thread at startup ---
if __name__ == "__main__":
    print("Starting SmartAgro server…")
    if AUTO_TRAINING:
        socketio.start_background_task(retrain_weather_model_daily)
    # Start auto-watering background thread
    socketio.start_background_task(auto_watering_worker)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)