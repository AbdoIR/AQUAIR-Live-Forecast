# AQUAIR PM2.5 Live Forecast + Alarm

This project trains a PM2.5 15-minute-ahead model and uses it in a live hatchery
alarm system. In deployment, predictions are made only from live sensor history:
the latest sensor row plus the previous 12 five-minute rows.

## Project Structure

```text
alarm_monitor.py          Telegram alarm loop for live CSV monitoring
model_interface.py        Live API server and shared prediction helpers
simulate_sensors.py       Local hatchery sensor simulator
telegram_alarm.py         Telegram Bot API sender
config.py                 Shared telegram.env loader
frontend/                 React dashboard HTML
dataset/                  Training/demo datasets
models/                   Saved model artifacts and local alarm state
telegram.env.example      Example local configuration
```

## Data Flow

1. Train and save the best model from `dataset/aquair_final.csv`.
2. The sensor system appends live readings to a CSV.
3. The monitor reads the latest 13 live rows.
4. The backend computes lag, rolling, cyclical, and interaction features.
5. The model predicts PM2.5 15 minutes ahead.
6. The alarm uses `max(current PM2.5, predicted PM2.5)` for safety.
7. Telegram is sent when that risk value reaches the dangerous threshold.

The live CSV must contain:

```text
timestamp(UTC+1),score,temp,humid,co2,voc,pm25,pm10
```

Sampling is assumed to be every 5 minutes.

## Train

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train in the notebook, then save the best model artifact to `models/best_model.pkl`.

## Live CSV Warm-Up

The alarm will not predict until at least 13 live rows exist:

- 12 previous readings for lag/rolling context
- 1 current reading

During warm-up it prints:

```text
Waiting for live history: X/13 rows available
```

This prevents the system from using old training data as fake live history.

## Simulate Hatchery Sensors

For local testing, use `simulate_sensors.py` to append realistic live readings to
`live_sensor.csv`.

Create 13 normal rows, enough for warm-up:

```powershell
python simulate_sensors.py --reset --scenario normal --rows 13
```

Check the monitor without sending Telegram:

```powershell
python alarm_monitor.py --dry-run
```

Append a high-pollution event:

```powershell
python simulate_sensors.py --scenario high --rows 3
python alarm_monitor.py --dry-run --cooldown-minutes 0
```

Append an unhealthy event:

```powershell
python simulate_sensors.py --scenario unhealthy --rows 3
python alarm_monitor.py --dry-run --cooldown-minutes 0
```

To mimic real streaming, append one row every few seconds while the monitor polls:

```powershell
python simulate_sensors.py --reset --scenario moderate --rows 30 --sleep-seconds 5
```

In another terminal:

```powershell
python alarm_monitor.py --dry-run --interval-seconds 5 --cooldown-minutes 0
```

Available scenarios are `normal`, `moderate`, `high`, and `unhealthy`.

## Telegram Setup

Create `telegram.env` in the project folder:

```text
TELEGRAM_BOT_TOKEN=123456789:your_bot_token
TELEGRAM_CHAT_ID=1481382195
HATCHERY_NAME=Azrou hatchery
ALARM_MIN_PM25=35
LIVE_SENSOR_CSV=C:\path\to\live_sensor.csv
```

You can also pass the live CSV explicitly with `--source`.

Test without sending:

```powershell
python alarm_monitor.py --source C:\path\to\live_sensor.csv --dry-run
```

Run one real check:

```powershell
python alarm_monitor.py --source C:\path\to\live_sensor.csv
```

Run continuously every five minutes:

```powershell
python alarm_monitor.py --source C:\path\to\live_sensor.csv --interval-seconds 300
```

Simulate an unhealthy forecast for Telegram testing:

```powershell
python alarm_monitor.py --source C:\path\to\live_sensor.csv --simulate-prediction 60 --cooldown-minutes 0
```

## Live Dashboard

Run the demo interface from the same live CSV:

```powershell
python model_interface.py --source C:\path\to\live_sensor.csv
```

Then open:

```text
http://127.0.0.1:8000
```

The browser demo can:

- add one `normal`, `moderate`, `high`, or `unhealthy` simulated sensor row
- reset `live_sensor.csv`
- visualize PM2.5 history
- show a PM2.5 risk gauge
- visualize CO2/VOC supporting trends
- show a latest sensor snapshot for PM2.5, PM10, CO2, and VOC
- show warning-band distribution across recent rows
- show PM2.5 momentum, where positive bars mean rising pollution
- show current risk drivers from PM2.5, PM10, CO2, and VOC
- refresh prediction automatically
- send a compact report to the AI consultation endpoint

You still need 13 rows before prediction is ready. The Telegram monitor reads the
same `live_sensor.csv`.

## Warning Bands

- `Normal / Good`: PM2.5 from 0 to below 12
- `Moderate`: PM2.5 from 12 to below 35
- `High Pollution`: PM2.5 from 35 to below 55
- `Unhealthy`: PM2.5 at or above 55

The Telegram threshold defaults to `35 ug/m3`, meaning `High Pollution` and
`Unhealthy` trigger alarms. Change this with `ALARM_MIN_PM25` or `--min-pm25`.

## Model Performance Ideas

- Retrain with raw current sensor features included: `pm25`, `pm10`, `co2`,
  `voc`, `temp`, `humid`, and `score`.
- Add richer live-history features: 15m/30m/1h/2h lags, rolling min/max/std,
  and short-term slope.
- Evaluate alarm performance, not only regression: recall, precision, and
  false-alarm rate for `High Pollution` and `Unhealthy`.
- Use chronological backtesting only; avoid random splits for time-series data.
- Keep an untouched final test period and report both forecast metrics and alarm
  metrics.
- Retrain periodically with new hatchery data to handle seasonality, sensor
  drift, and facility changes.

## NVIDIA LLM Consultation

Use `Analyze` in the dashboard to send a compact PM2.5 report to an NVIDIA
OpenAI-compatible endpoint and show the consultation result in the page.

Add this to `telegram.env`:

```text
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
```

The dashboard sends only necessary information: current warning status, latest
sensor snapshot, recent statistics, high-pollution events, and model context.
