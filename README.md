# AQUAIR PM2.5 Forecast + Alarm

Live PM2.5 warning system for a hatchery. The model predicts PM2.5 15 minutes ahead from live sensor history, then sends Telegram alerts when the current or predicted risk is dangerous.

## Main Features

- 15-minute PM2.5 forecasting
- Live CSV polling from sensor readings
- Warm-up rule: prediction starts after 13 live rows
- Conservative alarm logic: `risk_pm25 = max(current_pm25, predicted_pm25)`
- Telegram alerts for dangerous PM2.5
- React dashboard served by `model_interface.py`
- Sensor simulator for demo/testing
- AI consultation endpoint for dashboard analysis

## Project Files

```text
model_interface.py              API server + dashboard backend
frontend/react_dashboard.html   Dashboard UI
alarm_monitor.py                Telegram alarm monitor
telegram_alarm.py               Telegram sender
simulate_sensors.py             Local live sensor simulator
config.py                       telegram.env loader
training.ipynb                  Model training
model_comparison_plots.ipynb    Model comparison plots
preprocessing.ipynb             Data cleaning / exploration
feature_eng.ipynb               Feature engineering
dataset/                        Input datasets
models/                         Saved model + metadata
```

## Data Format

Live CSV columns:

```text
timestamp(UTC+1),score,temp,humid,co2,voc,pm25,pm10
```

Production inference uses only live CSV rows. It does not use old training rows as recent context.

## Install

```powershell
pip install -r requirements.txt
```

## Train Models

Open and run:

```text
training.ipynb
```

Models compared:

- XGBoost
- Random Forest
- LightGBM

The best model is saved to:

```text
models/best_model.pkl
models/best_model_metadata.json
models/model_comparison.json
```

To visualize model results, run:

```text
model_comparison_plots.ipynb
```

## Run Dashboard

```powershell
python model_interface.py
```

Open:

```text
http://127.0.0.1:8000
```

## Simulate Sensor Rows

Create enough rows for warm-up:

```powershell
python simulate_sensors.py --reset --scenario normal --rows 13
```

Add a dangerous event:

```powershell
python simulate_sensors.py --scenario high --rows 3
```

Available scenarios:

```text
normal, moderate, high, unhealthy
```

## Telegram Setup

Create `telegram.env`:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
HATCHERY_NAME=Azrou hatchery
ALARM_MIN_PM25=35
LIVE_SENSOR_CSV=live_sensor.csv
```

Dry-run check:

```powershell
python alarm_monitor.py --dry-run
```

Run continuous monitor:

```powershell
python alarm_monitor.py --interval-seconds 300
```

## Warning Bands

| PM2.5 ug/m3 | Level |
|---:|---|
| 0-12 | Normal / Good |
| 12-35 | Moderate |
| 35-55 | High Pollution |
| 55+ | Unhealthy |

Telegram alerts start at `35 ug/m3` by default.

## Optional AI Consultation

Add to `telegram.env` if using the dashboard Analyze button:

```text
NVIDIA_API_KEY=your_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
```
