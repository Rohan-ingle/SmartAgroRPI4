# Development

For Development consider the development repository at [Rohan-ingle/SmartAgroDev](https://github.com/Rohan-ingle/SmartAgroDev)

# SmartAgroRPI4

SmartAgroRPI4 integrates plant disease classification, animal intrusion detection, automated watering control with toggle option, and Gemini-based disease mitigation assistance. Initial training and development reside in the SmartAgroDev repository.

## Overview

- Plant disease classification via an EfficientNet model  
- Animal intrusion detection using **YOLOv11 n** (nano‑variant), optimized for edge efficiency and real-time performance
- Smart watering decision based on local XGBoost weather prediction and external forecast, with manual toggle to enable or disable automation  
- Gemini AI integration provides grounded disease information, treatment suggestions, and proactive mitigation advice

## Workflow

1. UI captures plant image → EfficientNet classifies disease  
2. Gemini AI delivers disease mitigation guidance  
3. YOLOv11 n monitors camera feed for animal intrusion  
4. Local XGBoost predicts short‑term weather and retrieves external forecast  
5. Watering automation logic (user toggle available) determines action  
6. UI displays disease classification, mitigation advice, intrusion alerts, weather info, and watering status


## Installation

```bash
git clone https://github.com/Rohan-ingle/SmartAgroRPI4.git
cd SmartAgroRPI4
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with the following format:

```env
# WeatherAPI Configuration
WEATHER_API_KEY=your_weatherapi_key_here
WEATHER_LOCATION = 'location'

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note:** 
- Get your WeatherAPI key from [WeatherAPI.com](https://www.weatherapi.com/)
- Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

```bash
cd SMARTAGRORPI4
python app.py
```
## UI

The UI will be available at ipaddr:5000 where ipaddr is the ip of the host machine


----

**Open to contributions**