import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from config import load_env_file
from simulate_sensors import HEADER, SCENARIOS, append_reading, ensure_header, make_reading, read_last_timestamp
from telegram_alarm import TelegramConfigError, send_telegram_message


BASE_DIR = Path(__file__).resolve().parent
FINAL_DATA_PATH = BASE_DIR / "dataset" / "aquair_final.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DEFAULT_LIVE_SOURCE = BASE_DIR / "live_sensor.csv"

TIMESTAMP_COL = "timestamp(UTC+1)"
TARGET_COL = "target_pm25_15m"
RAW_COLUMNS = ["score", "temp", "humid", "co2", "voc", "pm25", "pm10"]
FEATURE_KEYWORDS = ("lag", "sin", "cos", "rolling", "inter")
LOCAL_TZ = timezone(timedelta(hours=1))
DEFAULT_HISTORY_SIZE = 12
TELEGRAM_COOLDOWN_SECONDS = 30 * 60
TELEGRAM_STATE = {"last_sent_at": 0, "last_level": None, "last_timestamp": None}
LEVEL_RANK = {
    "Normal / Good": 0,
    "Moderate": 1,
    "High Pollution": 2,
    "Unhealthy": 3,
}


WARNING_LEVELS = [
    {"name": "Normal / Good", "min": 0.0, "max": 12.0, "color": "#6f9f87", "action": "Normal / good air condition."},
    {"name": "Moderate", "min": 12.0, "max": 35.0, "color": "#c7a85d", "action": "Moderate PM2.5 level. The facility should be monitored."},
    {"name": "High Pollution", "min": 35.0, "max": 55.0, "color": "#c98a67", "action": "High pollution event. This may be a possible risk indicator for the facility."},
    {"name": "Unhealthy", "min": 55.0, "max": None, "color": "#bd7474", "action": "Unhealthy air condition. Action is needed."},
]


class NumpyRidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.scale_ = None
        self.coef_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        x_scaled = (x - self.mean_) / self.scale_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        penalty = np.eye(x_design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        x_scaled = (x - self.mean_) / self.scale_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        return x_design @ self.coef_


@dataclass
class AppState:
    features: list[str]
    model_name: str
    model: object
    model_source: str
    live_source: Path
    history_size: int
    demo_mode: bool = True


def to_float(value, default=0.0):
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def parse_local_timestamp(value=None):
    if value is None or value == "" or pd.isna(value):
        return pd.Timestamp.now(tz=LOCAL_TZ)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.Timestamp.now(tz=LOCAL_TZ)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(LOCAL_TZ)
    return timestamp.tz_convert(LOCAL_TZ)


def normalize_timestamp_column(df):
    df = df.copy()
    df[TIMESTAMP_COL] = [parse_local_timestamp(value) for value in df[TIMESTAMP_COL]]
    return df


def make_model():
    try:
        from xgboost import XGBRegressor

        return (
            "XGBoost startup model",
            XGBRegressor(
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
                subsample=0.7,
                n_estimators=200,
                min_child_weight=1,
                max_depth=3,
                learning_rate=0.05,
                colsample_bytree=0.7,
            ),
        )
    except Exception:
        pass

    try:
        from sklearn.ensemble import RandomForestRegressor

        return (
            "Random Forest startup model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features="log2",
                max_depth=10,
            ),
        )
    except Exception:
        pass

    return "Ridge fallback", NumpyRidgeRegressor(alpha=1.0)


def load_model_state(live_source=None, history_size=DEFAULT_HISTORY_SIZE):
    load_env_file()
    if not FINAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training dataset: {FINAL_DATA_PATH}")

    df_final = normalize_timestamp_column(pd.read_csv(FINAL_DATA_PATH))
    features = [c for c in df_final.columns if any(k in c for k in FEATURE_KEYWORDS)]

    artifact = None
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as f:
                artifact = pickle.load(f)
        except Exception as exc:
            print(f"Could not load saved model at {MODEL_PATH}: {exc}")

    if artifact is not None:
        model = artifact["model"]
        model_name = artifact.get("model_name", type(model).__name__)
        features = artifact.get("features", features)
        model_source = str(MODEL_PATH)
    else:
        holdout_size = int(len(df_final) * 0.15)
        train_end = len(df_final) - holdout_size
        model_name, model = make_model()
        model.fit(df_final.loc[: train_end - 1, features], df_final.loc[: train_end - 1, TARGET_COL])
        model_source = "trained at startup; run train_and_save_model.py for a saved best model"

    source = Path(live_source or os.getenv("LIVE_SENSOR_CSV") or DEFAULT_LIVE_SOURCE)
    return AppState(
        features=features,
        model_name=model_name,
        model=model,
        model_source=model_source,
        live_source=source,
        history_size=history_size,
    )


def normalize_reading(reading, fallback_timestamp=None):
    timestamp = reading.get(TIMESTAMP_COL) or reading.get("timestamp") or fallback_timestamp
    output = {TIMESTAMP_COL: parse_local_timestamp(timestamp)}
    for col in RAW_COLUMNS:
        output[col] = to_float(reading.get(col), default=np.nan)
    return output


def load_live_rows(source_path):
    source = Path(source_path)
    if not source.exists():
        ensure_header(source)
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[TIMESTAMP_COL, *RAW_COLUMNS])

    missing = [col for col in [TIMESTAMP_COL, *RAW_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"Live sensor CSV is missing columns: {', '.join(missing)}")

    df = normalize_timestamp_column(df)
    df = df.dropna(subset=["temp", "humid", "co2", "pm25"])
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


def live_context(source_path, history_size=DEFAULT_HISTORY_SIZE):
    df = load_live_rows(source_path)
    required_rows = history_size + 1
    available_rows = len(df)
    latest_timestamp = str(df.iloc[-1][TIMESTAMP_COL]) if available_rows else None

    if available_rows < required_rows:
        return {
            "ready": False,
            "available_rows": available_rows,
            "required_rows": required_rows,
            "latest_timestamp": latest_timestamp,
            "message": f"Waiting for live history: {available_rows}/{required_rows} rows available",
            "expected_header": ",".join([TIMESTAMP_COL, *RAW_COLUMNS]),
        }

    rows = df.tail(required_rows).copy()
    rows[TIMESTAMP_COL] = rows[TIMESTAMP_COL].astype(str)
    readings = rows[[TIMESTAMP_COL, *RAW_COLUMNS]].to_dict(orient="records")
    return {
        "ready": True,
        "available_rows": available_rows,
        "required_rows": required_rows,
        "latest_timestamp": readings[-1][TIMESTAMP_COL],
        "recent_readings": readings[:-1],
        "current_reading": readings[-1],
        "message": f"{required_rows} rows available, ready",
    }


def history_payload(source_path, limit=120):
    df = load_live_rows(source_path)
    if not df.empty:
        latest = df[TIMESTAMP_COL].max()
        start = latest - pd.Timedelta(hours=24)
        df = df[df[TIMESTAMP_COL] >= start].copy()
        df = df.set_index(TIMESTAMP_COL)
        numeric_cols = [col for col in RAW_COLUMNS if col in df.columns]
        df = df[numeric_cols].resample("15min").mean().dropna(subset=["pm25"]).reset_index()
    if len(df) > limit:
        df = df.tail(limit)
    df = df.copy()
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype(str)
    return {
        "rows": df[[TIMESTAMP_COL, *RAW_COLUMNS]].to_dict(orient="records"),
        "count": len(df),
        "source": str(source_path),
        "range": "last 24h",
        "point_interval": "15min",
    }


def sensor_summary_payload(source_path):
    df = load_live_rows(source_path)
    if df.empty:
        return {"ready": False, "count": 0}

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    recent = df.tail(12)
    return {
        "ready": True,
        "count": len(df),
        "latest_timestamp": str(latest[TIMESTAMP_COL]),
        "latest": {col: to_float(latest[col]) for col in RAW_COLUMNS},
        "delta_pm25": to_float(latest["pm25"]) - to_float(previous["pm25"]),
        "avg_pm25_1h": to_float(recent["pm25"].mean()),
        "max_pm25_1h": to_float(recent["pm25"].max()),
        "avg_co2_1h": to_float(recent["co2"].mean()),
        "avg_voc_1h": to_float(recent["voc"].mean()),
        "level": warning_level(to_float(latest["pm25"])),
    }


def build_feature_row(recent_readings, current_reading, features):
    frame = pd.DataFrame([*[normalize_reading(row) for row in recent_readings], normalize_reading(current_reading)])
    frame[TIMESTAMP_COL] = [parse_local_timestamp(value) for value in frame[TIMESTAMP_COL]]
    frame = frame.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if len(frame) < 3:
        raise ValueError("At least two previous 5-minute readings are required for lag features.")

    frame["hour"] = frame[TIMESTAMP_COL].dt.hour
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    for lag in [1, 2]:
        frame[f"pm25_lag_{lag}"] = frame["pm25"].shift(lag)
        frame[f"co2_lag_{lag}"] = frame["co2"].shift(lag)
    frame["pm25_rolling_mean_1h"] = frame["pm25"].rolling(window=12, min_periods=1).mean()
    frame["temp_rolling_mean_30m"] = frame["temp"].rolling(window=6, min_periods=1).mean()
    frame["pm25_volatility_1h"] = frame["pm25"].rolling(window=12, min_periods=2).std().fillna(0)
    frame["humid_temp_inter"] = frame["humid"] * frame["temp"]

    feature_row = frame.iloc[-1]
    missing = [feature for feature in features if pd.isna(feature_row.get(feature))]
    if missing:
        raise ValueError(f"Could not compute required features: {', '.join(missing)}")

    return {feature: to_float(feature_row[feature]) for feature in features}, frame


def warning_level(pm25_value):
    for level in WARNING_LEVELS:
        max_value = level["max"]
        if level["min"] <= pm25_value and (max_value is None or pm25_value < max_value):
            return level
    return WARNING_LEVELS[-1]


def trend_message(frame, predicted_pm25):
    current_pm25 = to_float(frame.iloc[-1]["pm25"])
    previous_pm25 = to_float(frame.iloc[-2]["pm25"])
    delta_future = predicted_pm25 - current_pm25
    if delta_future >= 3:
        direction = "rising"
    elif delta_future <= -3:
        direction = "falling"
    else:
        direction = "stable"
    return {
        "current_pm25": current_pm25,
        "previous_pm25": previous_pm25,
        "delta_from_previous": current_pm25 - previous_pm25,
        "delta_next_15m": delta_future,
        "direction": direction,
    }


def warning_payload(payload, state):
    features, frame = build_feature_row(payload.get("recent_readings") or [], payload.get("current_reading") or {}, state.features)
    prediction = float(state.model.predict(pd.DataFrame([features], columns=state.features))[0])
    trend = trend_message(frame, prediction)
    risk_pm25 = max(prediction, trend["current_pm25"])
    risk_source = "current reading" if trend["current_pm25"] >= prediction else "15-minute forecast"
    return {
        "prediction": prediction,
        "risk_pm25": risk_pm25,
        "risk_source": risk_source,
        "level": warning_level(risk_pm25),
        "trend": trend,
        "features_used": features,
        "model_name": state.model_name,
        "model_source": state.model_source,
    }


def live_warning_payload(state):
    context = live_context(state.live_source, state.history_size)
    if not context["ready"]:
        return {
            **context,
            "source": str(state.live_source),
            "demo_mode": state.demo_mode,
            "model_name": state.model_name,
            "model_source": state.model_source,
        }
    warning = warning_payload(
        {"recent_readings": context["recent_readings"], "current_reading": context["current_reading"]},
        state,
    )
    return {**context, **warning, "source": str(state.live_source), "demo_mode": state.demo_mode}


def format_dashboard_telegram_message(result):
    level = result["level"]
    trend = result["trend"]
    return "\n".join(
        [
            "AQUAIR PM2.5 Alert",
            "",
            f"Status: {level['name']}",
            f"Risk PM2.5: {result['risk_pm25']:.1f} ug/m3",
            f"Basis: {result['risk_source']}",
            "",
            f"Current: {trend['current_pm25']:.1f} ug/m3",
            f"Forecast +15 min: {result['prediction']:.1f} ug/m3",
            f"Trend: {trend['direction']} ({trend['delta_next_15m']:+.1f})",
            "",
            f"Action: {level['action']}",
            f"Time: {result['latest_timestamp']}",
        ]
    )


def maybe_send_dashboard_alarm(result):
    if not result.get("ready"):
        return {"sent": False, "reason": "not ready"}

    threshold = float(os.getenv("ALARM_MIN_PM25", "35"))
    risk_pm25 = result.get("risk_pm25", 0)
    if risk_pm25 < threshold:
        TELEGRAM_STATE["last_sent_at"] = 0
        TELEGRAM_STATE["last_level"] = None
        TELEGRAM_STATE["last_timestamp"] = None
        return {"sent": False, "reason": "below threshold"}

    now = datetime.now(tz=LOCAL_TZ).timestamp()
    level_name = result["level"]["name"]
    latest_timestamp = result.get("latest_timestamp")
    level_got_worse = LEVEL_RANK.get(level_name, 0) > LEVEL_RANK.get(TELEGRAM_STATE.get("last_level"), -1)
    cooldown_elapsed = now - TELEGRAM_STATE.get("last_sent_at", 0) >= TELEGRAM_COOLDOWN_SECONDS
    same_timestamp = latest_timestamp == TELEGRAM_STATE.get("last_timestamp")

    if same_timestamp and not level_got_worse:
        return {"sent": False, "reason": "already sent for this row"}

    if not cooldown_elapsed and not level_got_worse:
        return {"sent": False, "reason": "cooldown active"}

    try:
        send_telegram_message(format_dashboard_telegram_message(result))
    except TelegramConfigError as exc:
        return {"sent": False, "reason": str(exc)}
    except Exception as exc:
        return {"sent": False, "reason": f"Telegram send failed: {exc}"}

    TELEGRAM_STATE.update(
        {
            "last_sent_at": now,
            "last_level": level_name,
            "last_timestamp": latest_timestamp,
        }
    )
    return {"sent": True, "reason": "sent"}


def next_demo_timestamp(source_path):
    ensure_header(source_path)
    last_timestamp = read_last_timestamp(source_path)
    if last_timestamp is None:
        return datetime.now(tz=LOCAL_TZ).replace(second=0, microsecond=0)
    return last_timestamp + timedelta(minutes=5)


def add_demo_row(source_path, scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    ensure_header(source_path)
    timestamp = next_demo_timestamp(source_path)
    reading = make_reading(timestamp, step=0, scenario=scenario)
    append_reading(source_path, reading)
    return reading


def reset_live_source(source_path):
    ensure_header(source_path, reset=True)
    return {"source": str(source_path), "message": "live_sensor.csv rows reset"}


def json_response(handler, payload, status=200):
    body = json.dumps(payload, default=float, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AQUAIR PM2.5 Demo</title>
  <style>
    :root {
      font-family: Inter, ui-sans-serif, system-ui, Segoe UI, sans-serif;
      background: #eef3f8;
      color: #132033;
      --blue: #5b7fc7;
      --green: #6f9f87;
      --yellow: #c7a85d;
      --orange: #c98a67;
      --red: #bd7474;
      --ink-soft: #5d6a7f;
      --line: #d9e2ee;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header {
      padding: 24px min(5vw, 64px);
      background: #ffffff;
      color: #132033;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 6px; font-size: clamp(24px, 3vw, 38px); }
    main { padding: 18px min(5vw, 64px); display: grid; gap: 10px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    button {
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 11px 14px;
      font-weight: 850;
      cursor: pointer;
      color: #fff;
      transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10);
    }
    button:hover { transform: translateY(-1px); filter: brightness(1.02); }
    button.normal { background: #6f9f87; }
    button.moderate { background: #c7a85d; }
    button.high { background: #c98a67; }
    button.unhealthy { background: #bd7474; }
    button.secondary { background: #e8eef9; color: #1b2a44; border-color: #cbd7e7; box-shadow: none; }
    button.danger { background: #334155; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin-top: 12px; }
    .metric { border: 1px solid #e0e7f1; border-radius: 8px; padding: 14px; background: linear-gradient(180deg, #fff, #f8fbff); }
    .metric span { display: block; color: #617089; font-size: 13px; }
    .metric strong { display: block; margin-top: 4px; font-size: 24px; }
    .alert { border-radius: 8px; padding: 18px; color: #fff; background: #64748b; }
    .alert strong { display: block; font-size: clamp(28px, 4vw, 46px); line-height: 1; }
    .alert span { display: block; margin-top: 8px; font-weight: 700; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
    .legend span { border-radius: 999px; padding: 6px 10px; color: #4b5563; font-weight: 500; font-size: 13px; }
    .charts { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 8px; }
    canvas { width: 100%; height: 300px; border: 1px solid #e0e7f1; border-radius: 8px; background: #fff; display: block; }
    #gaugeChart { height: 130px; margin-top: 10px; }
    .subtle, .fineprint { color: var(--ink-soft); }
    .notice { background: #f4f7fb; border: 1px solid #dbe4ef; border-radius: 8px; padding: 10px 12px; color: #475569; }
    header .subtle { color: var(--ink-soft); }
    @media (max-width: 900px) { .metric-grid { grid-template-columns: 1fr; } }
    @media (max-width: 1100px) { .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>AQUAIR PM2.5 Demo Interface</h1>
    <p class="subtle">Simulate hatchery sensor rows, visualize PM2.5, then run the 15-minute warning prediction.</p>
  </header>
  <main>
    <section>
      <h2>Sensor Simulation Controls</h2>
      <div class="controls">
        <button class="normal" data-scenario="normal">Add Normal Row</button>
        <button class="moderate" data-scenario="moderate">Add Moderate Row</button>
        <button class="high" data-scenario="high">Add High Row</button>
        <button class="unhealthy" data-scenario="unhealthy">Add Unhealthy Row</button>
        <button id="reset" class="danger">Reset Rows</button>
        <button id="predict" class="secondary">Predict</button>
      </div>
      <p id="status" class="fineprint">Ready.</p>
    </section>

    <section>
      <div id="alert" class="alert">
        <strong id="level">Waiting</strong>
        <span id="action">Add at least 13 rows, then press Predict.</span>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>PM2.5 used for warning</span><strong id="riskPm25">-</strong></div>
        <div class="metric"><span>Predicted PM2.5 in 15m</span><strong id="prediction">-</strong></div>
        <div class="metric"><span>Current PM2.5</span><strong id="currentPm25">-</strong></div>
        <div class="metric"><span>Live history</span><strong id="history">-</strong></div>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>Current CO2</span><strong id="currentCo2">-</strong></div>
        <div class="metric"><span>Current VOC</span><strong id="currentVoc">-</strong></div>
        <div class="metric"><span>Current PM10</span><strong id="currentPm10">-</strong></div>
        <div class="metric"><span>1h Avg / Max PM2.5</span><strong id="pm25Stats">-</strong></div>
      </div>
      <p class="fineprint">Latest timestamp: <strong id="timestamp">-</strong></p>
      <p class="fineprint">Warning basis: <strong id="riskSource">-</strong>. Trend: <strong id="trend">-</strong>.</p>
      <canvas id="gaugeChart" width="1800" height="240"></canvas>
    </section>

    <section>
      <h2>Dashboard Visuals</h2>
      <p class="fineprint">Compact views for explaining air quality level, sensor context, and short-term movement. Charts show the latest 24 hours with one point every 15 minutes.</p>
      <div class="charts">
        <div>
          <h3>PM2.5 History</h3>
          <canvas id="chart" width="1800" height="560"></canvas>
          <p class="fineprint">Each point is a 15-minute average from the latest 24 hours, so several raw 5-minute rows become one point.</p>
          <div class="legend">
            <span style="background:#d8eadf">0-12 Normal</span>
            <span style="background:#efe2bd">12-35 Moderate</span>
            <span style="background:#f0d2c3">35-55 High</span>
            <span style="background:#ead0d0">55+ Unhealthy</span>
          </div>
        </div>
        <div>
          <h3>Current Snapshot</h3>
          <canvas id="barChart" width="900" height="560"></canvas>
        </div>
        <div>
          <h3>CO2 / VOC Trend</h3>
          <canvas id="gasChart" width="1800" height="560"></canvas>
        </div>
        <div>
          <h3>Warning Mix</h3>
          <canvas id="bandChart" width="900" height="560"></canvas>
        </div>
        <div>
          <h3>PM2.5 Momentum</h3>
          <canvas id="momentumChart" width="900" height="560"></canvas>
        </div>
      </div>
    </section>
  </main>

  <script>
    const fmt = (n) => Number(n).toFixed(2);

    function prepareCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, rect.width || 800);
      const height = Math.max(120, rect.height || 300);
      const pixelWidth = Math.floor(width * ratio);
      const pixelHeight = Math.floor(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.font = "13px Inter, Segoe UI, sans-serif";
      return { ctx, w: width, h: height };
    }

    function clearCanvas(ctx, w, h) {
      ctx.save();
      ctx.setLineDash([]);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.restore();
    }

    function pointTimeLabel(row) {
      const raw = row["timestamp(UTC+1)"] || row.timestamp || "";
      const date = new Date(raw);
      if (!Number.isNaN(date.valueOf())) {
        return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      }
      const match = String(raw).match(/(\\d{2}:\\d{2})/);
      return match ? match[1] : "";
    }

    function drawTimeTicks(ctx, rows, xFor, y) {
      if (!rows.length) return;
      const maxLabels = 6;
      const step = Math.max(1, Math.ceil(rows.length / maxLabels));
      const labelIndexes = [];
      rows.forEach((row, index) => {
        const isLast = index === rows.length - 1;
        if (index % step === 0 || isLast) labelIndexes.push(index);
      });
      if (labelIndexes.length > 1) {
        const last = labelIndexes[labelIndexes.length - 1];
        const previous = labelIndexes[labelIndexes.length - 2];
        if (xFor(last) - xFor(previous) < 58) {
          labelIndexes.splice(labelIndexes.length - 2, 1);
        }
      }

      ctx.save();
      ctx.fillStyle = "#64748b";
      ctx.textAlign = "center";
      ctx.font = "12px Inter, Segoe UI, sans-serif";
      labelIndexes.forEach((index) => {
        const row = rows[index];
        ctx.fillText(pointTimeLabel(row), xFor(index), y);
      });
      ctx.restore();
    }

    function redrawCleanYAxis(ctx, minX, top, bottom, maxY) {
      const step = maxY <= 30 ? 5 : maxY <= 80 ? 10 : 20;
      const yFor = (v) => bottom - (v / maxY) * (bottom - top);
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, top - 18, minX - 8, bottom - top + 36);
      ctx.fillStyle = "#64748b";
      ctx.font = "12px Inter, Segoe UI, sans-serif";
      for (let tick = 0; tick <= maxY; tick += step) {
        const y = yFor(tick);
        ctx.fillText(String(tick), 18, y + 4);
      }
      ctx.restore();
    }

    function drawPm25ThresholdLines(ctx, minX, maxX, top, bottom, maxY) {
      const yFor = (v) => bottom - (v / maxY) * (bottom - top);
      ctx.save();
      ctx.strokeStyle = "#b56b7a";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      [12, 35, 55].forEach((threshold) => {
        if (threshold > maxY) return;
        const y = yFor(threshold);
        ctx.beginPath();
        ctx.moveTo(minX, y);
        ctx.lineTo(maxX, y);
        ctx.stroke();
      });
      ctx.restore();
    }

    function nicePm25Scale(values) {
      const maxValue = Math.max(0, ...values);
      const maxY = Math.max(12, Math.ceil((maxValue + 5) / 5) * 5);
      const step = maxY <= 20 ? 5 : maxY <= 60 ? 10 : 20;
      return { maxY, step };
    }

    function drawYAxis(ctx, minX, maxX, top, bottom, maxY, step) {
      const yFor = (v) => bottom - (v / maxY) * (bottom - top);
      const minLabelGap = 22;
      let lastLabelY = Infinity;
      ctx.save();
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#64748b";
      ctx.font = "12px Inter, Segoe UI, sans-serif";
      for (let tick = 0; tick <= maxY; tick += step) {
        const y = yFor(tick);
        ctx.beginPath();
        ctx.moveTo(minX, y);
        ctx.lineTo(maxX, y);
        ctx.stroke();
        if (Math.abs(y - lastLabelY) >= minLabelGap) {
          ctx.fillText(String(tick), 18, y + 4);
          lastLabelY = y;
        }
      }

      ctx.restore();
      return yFor;
    }

    async function jsonFetch(url, options) {
      const res = await fetch(url, options);
      const text = await res.text();
      const data = text ? JSON.parse(text) : {};
      if (!res.ok) throw new Error(data.error || text);
      return data;
    }

    function setStatus(text) {
      document.getElementById("status").textContent = text;
    }

    function renderPrediction(data) {
      const available = data.available_rows ?? 0;
      const required = data.required_rows ?? 13;
      document.getElementById("timestamp").textContent = data.latest_timestamp || "-";
      document.getElementById("history").textContent = `${available}/${required}`;

      const alert = document.getElementById("alert");
      if (!data.ready) {
        alert.style.background = "#64748b";
        document.getElementById("level").textContent = "Warming up";
        document.getElementById("action").textContent = data.message;
        ["riskPm25", "prediction", "currentPm25", "riskSource", "trend"].forEach(id => document.getElementById(id).textContent = "-");
        drawGauge(null);
        return;
      }

      alert.style.background = data.level.color;
      document.getElementById("level").textContent = data.level.name;
      document.getElementById("action").textContent = data.level.action;
      document.getElementById("riskPm25").textContent = fmt(data.risk_pm25);
      document.getElementById("prediction").textContent = fmt(data.prediction);
      document.getElementById("currentPm25").textContent = fmt(data.trend.current_pm25);
      document.getElementById("riskSource").textContent = data.risk_source;
      document.getElementById("trend").textContent = data.trend.direction;
      drawGauge(data.risk_pm25);
    }

    function drawGauge(value) {
      const canvas = document.getElementById("gaugeChart");
      const { ctx, w, h } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      const left = 54;
      const right = w - 54;
      const y = 54;
      const barH = 24;
      const max = 80;
      const bands = [
        [0, 12, "#2f7d5c", "Normal"],
        [12, 35, "#b58b2a", "Moderate"],
        [35, 55, "#c26a3a", "High"],
        [55, 80, "#a94444", "Unhealthy"],
      ];

      bands.forEach(([start, end, color, label]) => {
        const x = left + (start / max) * (right - left);
        const width = ((end - start) / max) * (right - left);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, width, barH);
        ctx.fillStyle = "#132033";
        ctx.fillText(label, x + 4, y + 48);
      });

      ctx.fillStyle = "#64748b";
      ctx.fillText("PM2.5 risk gauge (ug/m3)", left, 24);

      if (value === null || Number.isNaN(Number(value))) {
        ctx.fillText("Waiting for prediction", left, y - 10);
        return;
      }

      const clipped = Math.max(0, Math.min(max, Number(value)));
      const x = left + (clipped / max) * (right - left);
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x, y - 12);
      ctx.lineTo(x, y + barH + 12);
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
      ctx.fillText(`${fmt(value)} ug/m3`, Math.min(x + 8, right - 90), y - 12);
    }

    function renderSummary(data) {
      if (!data.ready) {
        ["currentCo2", "currentVoc", "currentPm10", "pm25Stats"].forEach(id => document.getElementById(id).textContent = "-");
        return;
      }
      document.getElementById("currentCo2").textContent = fmt(data.latest.co2);
      document.getElementById("currentVoc").textContent = fmt(data.latest.voc);
      document.getElementById("currentPm10").textContent = fmt(data.latest.pm10);
      document.getElementById("pm25Stats").textContent = `${fmt(data.avg_pm25_1h)} / ${fmt(data.max_pm25_1h)}`;
      document.getElementById("currentPm25").textContent = fmt(data.latest.pm25);
      document.getElementById("timestamp").textContent = data.latest_timestamp;
    }

    function drawChart(rows) {
      const canvas = document.getElementById("chart");
      const { ctx, w, h } = prepareCanvas(canvas);
      clearCanvas(ctx, w, h);

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      [12, 35, 55].forEach(threshold => {
        const maxY = Math.max(70, ...rows.map(r => Number(r.pm25 || 0)));
        const y = h - 36 - (threshold / maxY) * (h - 70);
        ctx.beginPath();
        ctx.moveTo(54, y);
        ctx.lineTo(w - 20, y);
        ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.fillText(`${threshold}`, 16, y + 4);
      });

      if (!rows.length) {
        ctx.fillStyle = "#64748b";
        ctx.fillText("No rows yet. Add simulated sensor rows.", 54, 170);
        return;
      }

      const values = rows.map(r => Number(r.pm25));
      const { maxY, step: yStep } = nicePm25Scale(values);
      const minX = 84;
      const maxX = w - 36;
      const top = 28;
      const bottom = h - 58;
      const xFor = (i) => rows.length === 1 ? minX : minX + (i / (rows.length - 1)) * (maxX - minX);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(minX - 1, top - 1, maxX - minX + 2, bottom - top + 2);
      const yFor = drawYAxis(ctx, minX, maxX, top, bottom, maxY, yStep);

      ctx.strokeStyle = "#5b7fc7";
      ctx.lineWidth = 3;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = xFor(i);
        const y = yFor(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#5b7fc7";
      values.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(xFor(i), yFor(v), 4, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.fillStyle = "#132033";
      drawTimeTicks(ctx, rows, xFor, h - 34);
      ctx.fillText(`15-min points: ${rows.length} | Latest PM2.5: ${fmt(values[values.length - 1])}`, 54, h - 12);
      redrawCleanYAxis(ctx, minX, top, bottom, maxY);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(minX - 1, top - 1, maxX - minX + 2, bottom - top + 2);
      const cleanYFor = drawYAxis(ctx, minX, maxX, top, bottom, maxY, yStep);
      drawPm25ThresholdLines(ctx, minX, maxX, top, bottom, maxY);
      ctx.strokeStyle = "#5b7fc7";
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = xFor(i);
        const y = cleanYFor(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = "#5b7fc7";
      values.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(xFor(i), cleanYFor(v), 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function drawGasChart(rows) {
      const canvas = document.getElementById("gasChart");
      const { ctx, w, h } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      if (!rows.length) {
        ctx.fillStyle = "#64748b";
        ctx.fillText("No rows yet.", 40, 160);
        return;
      }

      const co2 = rows.map(r => Number(r.co2 || 0));
      const voc = rows.map(r => Number(r.voc || 0));
      const vocScaled = voc.map(v => v * 4);
      const maxY = Math.max(1000, ...co2, ...vocScaled);
      const minX = 52;
      const maxX = w - 24;
      const top = 24;
      const bottom = h - 42;
      const xFor = (i) => rows.length === 1 ? minX : minX + (i / (rows.length - 1)) * (maxX - minX);
      const yFor = (v) => bottom - (v / maxY) * (bottom - top);

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      [400, 800, 1200].forEach(value => {
        const y = yFor(value);
        ctx.beginPath();
        ctx.moveTo(minX, y);
        ctx.lineTo(maxX, y);
        ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.fillText(`${value}`, 12, y + 4);
      });

      function line(values, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        values.forEach((v, i) => {
          const x = xFor(i);
          const y = yFor(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      line(co2, "#6f9f87");
      line(vocScaled, "#8d95a3");
      ctx.fillStyle = "#6f9f87";
      ctx.fillText("CO2", minX, h - 14);
      ctx.fillStyle = "#8d95a3";
      ctx.fillText("VOC x4", minX + 52, h - 14);
      drawTimeTicks(ctx, rows, xFor, h - 30);
    }

    function drawBarChart(summary) {
      const canvas = document.getElementById("barChart");
      const { ctx, w, h } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      if (!summary.ready) {
        ctx.fillStyle = "#64748b";
        ctx.fillText("No latest reading yet.", 40, 160);
        return;
      }

      const items = [
        ["PM2.5", summary.latest.pm25, 70, "#5b7fc7"],
        ["PM10", summary.latest.pm10, 100, "#8d95a3"],
        ["CO2", summary.latest.co2, 1400, "#6f9f87"],
        ["VOC", summary.latest.voc, 350, "#9aa1ad"],
      ];

      const left = 72;
      const barH = 34;
      const gap = 32;
      items.forEach(([label, value, scale, color], i) => {
        const y = 42 + i * (barH + gap);
        const width = Math.min(1, value / scale) * (w - 160);
        ctx.fillStyle = "#eef2f7";
        ctx.fillRect(left, y, w - 160, barH);
        ctx.fillStyle = color;
        ctx.fillRect(left, y, width, barH);
        ctx.fillStyle = "#132033";
        ctx.fillText(label, 20, y + 22);
        ctx.fillText(fmt(value), left + width + 8, y + 22);
      });
    }

    function bandForPm25(value) {
      if (value < 12) return "Normal";
      if (value < 35) return "Moderate";
      if (value < 55) return "High";
      return "Unhealthy";
    }

    function drawBandChart(rows) {
      const canvas = document.getElementById("bandChart");
      const { ctx, w, h } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      const bands = [
        ["Normal", "#6f9f87"],
        ["Moderate", "#c7a85d"],
        ["High", "#c98a67"],
        ["Unhealthy", "#bd7474"],
      ];
      const counts = Object.fromEntries(bands.map(([name]) => [name, 0]));
      rows.forEach(row => counts[bandForPm25(Number(row.pm25 || 0))] += 1);
      const total = Math.max(1, rows.length);
      const maxCount = Math.max(1, ...Object.values(counts));

      if (!rows.length) {
        ctx.fillStyle = "#64748b";
        ctx.fillText("No history yet.", 40, 160);
        return;
      }

      const left = 110;
      const labelX = w - 132;
      const barMax = Math.max(120, labelX - left - 16);
      const barH = 38;
      const gap = 28;
      bands.forEach(([name, color], i) => {
        const y = 42 + i * (barH + gap);
        const count = counts[name];
        const width = (count / maxCount) * barMax;
        const pct = Math.round((count / total) * 100);
        ctx.fillStyle = "#eef2f7";
        ctx.fillRect(left, y, barMax, barH);
        ctx.fillStyle = color;
        ctx.fillRect(left, y, width, barH);
        ctx.fillStyle = "#132033";
        ctx.fillText(name, 24, y + 24);
        ctx.fillText(`${count} points`, labelX, y + 17);
        ctx.fillText(`${pct}%`, labelX, y + 33);
      });
    }

    function drawMomentumChart(rows) {
      const canvas = document.getElementById("momentumChart");
      const { ctx, w, h } = prepareCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      if (rows.length < 2) {
        ctx.fillStyle = "#64748b";
        ctx.fillText("Need at least 2 rows for momentum.", 40, 160);
        return;
      }

      const recent = rows.slice(-16);
      const deltas = [];
      for (let i = 1; i < recent.length; i++) {
        deltas.push(Number(recent[i].pm25 || 0) - Number(recent[i - 1].pm25 || 0));
      }

      const maxAbs = Math.max(3, ...deltas.map(v => Math.abs(v)));
      const mid = h / 2;
      const left = 32;
      const right = w - 24;
      const gap = 6;
      const barW = (right - left - gap * (deltas.length - 1)) / deltas.length;

      ctx.strokeStyle = "#cbd5e1";
      ctx.beginPath();
      ctx.moveTo(left, mid);
      ctx.lineTo(right, mid);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.fillText("rising", 34, 26);
      ctx.fillText("falling", 34, h - 14);

      deltas.forEach((delta, i) => {
        const x = left + i * (barW + gap);
        const height = Math.abs(delta) / maxAbs * (h / 2 - 40);
        const y = delta >= 0 ? mid - height : mid;
        ctx.fillStyle = delta >= 0 ? "#c98a67" : "#6f9f87";
        ctx.fillRect(x, y, barW, height);
      });

      const latest = deltas[deltas.length - 1];
      ctx.fillStyle = "#132033";
      ctx.fillText(`Latest change: ${latest >= 0 ? "+" : ""}${fmt(latest)} ug/m3`, 34, h - 34);
    }

    async function refreshHistory() {
      const data = await jsonFetch("/api/history");
      drawChart(data.rows);
      drawGasChart(data.rows);
      drawBandChart(data.rows);
      drawMomentumChart(data.rows);
      return data;
    }

    async function refreshSummary() {
      const data = await jsonFetch("/api/summary");
      renderSummary(data);
      drawBarChart(data);
      return data;
    }

    async function predict() {
      const results = await Promise.allSettled([
        jsonFetch("/api/live"),
        refreshHistory(),
        refreshSummary()
      ]);

      if (results[0].status === "fulfilled") {
        renderPrediction(results[0].value);
        if (results[0].value.telegram) {
          const telegram = results[0].value.telegram;
          setStatus(telegram.sent ? "Prediction refreshed. Telegram alert sent." : `Prediction refreshed. Telegram: ${telegram.reason}.`);
        }
      } else {
        throw results[0].reason;
      }

      const failed = results.find(result => result.status === "rejected");
      if (failed) throw failed.reason;
    }

    document.querySelectorAll("[data-scenario]").forEach(button => {
      button.onclick = async () => {
        const scenario = button.getAttribute("data-scenario");
        const data = await jsonFetch(`/api/demo/add?scenario=${scenario}`, { method: "POST" });
        setStatus(`Added ${scenario} row at ${data.row["timestamp(UTC+1)"]} with PM2.5=${data.row.pm25}`);
        try {
          await predict();
        } catch (err) {
          setStatus(`Added row, but refresh failed: ${err.message}`);
        }
      };
    });

    document.getElementById("reset").onclick = async () => {
      await jsonFetch("/api/demo/reset", { method: "POST" });
      setStatus("live_sensor.csv reset. Add 13 rows before prediction.");
      try {
        await predict();
      } catch (err) {
        setStatus(`Reset complete, but refresh failed: ${err.message}`);
      }
    };

    document.getElementById("predict").onclick = async () => {
      try {
        await predict();
      } catch (err) {
        setStatus(`Prediction failed: ${err.message}`);
      }
    };

    predict().catch(err => setStatus(`Initial refresh failed: ${err.message}`));
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/live":
            try:
                payload = live_warning_payload(STATE)
                payload["telegram"] = maybe_send_dashboard_alarm(payload)
                json_response(self, payload)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/history":
            try:
                json_response(self, history_payload(STATE.live_source))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/summary":
            try:
                json_response(self, sensor_summary_payload(STATE.live_source))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        json_response(self, {"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/demo/reset":
                json_response(self, reset_live_source(STATE.live_source))
                return

            if parsed.path == "/api/demo/add":
                query = parse_qs(parsed.query)
                scenario = query.get("scenario", ["normal"])[0]
                row = add_demo_row(STATE.live_source, scenario)
                json_response(self, {"row": row, "source": str(STATE.live_source)})
                return

            json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=400)

    def log_message(self, format, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AQUAIR PM2.5 demo interface.")
    parser.add_argument("--source", type=Path, default=None, help="Live sensor CSV path.")
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    global STATE
    args = parse_args()
    STATE = load_model_state(args.source, args.history_size)
    ensure_header(STATE.live_source)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving AQUAIR PM2.5 demo interface at http://{args.host}:{args.port}")
    print(f"Live source CSV: {STATE.live_source}")
    server.serve_forever()


STATE = load_model_state()


if __name__ == "__main__":
    main()
