import argparse
import json
import math
import os
import pickle
import urllib.error
import urllib.request
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
REACT_DASHBOARD_PATH = BASE_DIR / "frontend" / "react_dashboard.html"

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
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


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


def history_payload(source_path, limit=160, range_hours=24, interval="15min"):
    df = load_live_rows(source_path)
    if not df.empty:
        latest = df[TIMESTAMP_COL].max()
        start = latest - pd.Timedelta(hours=range_hours)
        df = df[df[TIMESTAMP_COL] >= start].copy()
        df = df.set_index(TIMESTAMP_COL)
        numeric_cols = [col for col in RAW_COLUMNS if col in df.columns]
        df = df[numeric_cols].resample(interval).mean().dropna(subset=["pm25"]).reset_index()
    if len(df) > limit:
        df = df.tail(limit)
    df = df.copy()
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype(str)
    return {
        "rows": df[[TIMESTAMP_COL, *RAW_COLUMNS]].to_dict(orient="records"),
        "count": len(df),
        "source": str(source_path),
        "range": f"last {range_hours}h",
        "point_interval": interval,
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


def report_payload(state, range_hours=24, interval="15min"):
    live = live_warning_payload(state)
    history = history_payload(state.live_source, range_hours=range_hours, interval=interval)
    summary = sensor_summary_payload(state.live_source)
    rows = history["rows"]

    band_counts = {"Normal / Good": 0, "Moderate": 0, "High Pollution": 0, "Unhealthy": 0}
    for row in rows:
        level = warning_level(to_float(row.get("pm25")))["name"]
        band_counts[level] += 1

    events = []
    current_event = None
    for row in rows:
        pm25 = to_float(row.get("pm25"))
        level = warning_level(pm25)
        is_event = level["name"] in ["High Pollution", "Unhealthy"]
        timestamp = row.get(TIMESTAMP_COL)
        if is_event and current_event is None:
            current_event = {"level": level["name"], "start": timestamp, "end": timestamp, "max_pm25": pm25}
        elif is_event and current_event is not None:
            current_event["end"] = timestamp
            current_event["max_pm25"] = max(current_event["max_pm25"], pm25)
            if level["name"] == "Unhealthy":
                current_event["level"] = "Unhealthy"
        elif not is_event and current_event is not None:
            events.append(current_event)
            current_event = None
    if current_event is not None:
        events.append(current_event)

    return {
        "facility": os.getenv("HATCHERY_NAME", "Azrou hatchery"),
        "generated_at": datetime.now(tz=LOCAL_TZ).isoformat(timespec="seconds"),
        "source": str(state.live_source),
        "time_range": history["range"],
        "aggregation": history["point_interval"],
        "current_status": {
            "ready": live.get("ready", False),
            "level": live.get("level", {}).get("name"),
            "risk_pm25": live.get("risk_pm25"),
            "risk_source": live.get("risk_source"),
            "current_pm25": live.get("trend", {}).get("current_pm25"),
            "predicted_pm25_15m": live.get("prediction"),
            "trend": live.get("trend", {}).get("direction"),
            "latest_timestamp": live.get("latest_timestamp"),
        },
        "sensor_snapshot": summary.get("latest", {}),
        "recent_statistics": {
            "points": history["count"],
            "band_counts": band_counts,
            "avg_pm25_1h": summary.get("avg_pm25_1h"),
            "max_pm25_1h": summary.get("max_pm25_1h"),
            "avg_co2_1h": summary.get("avg_co2_1h"),
            "avg_voc_1h": summary.get("avg_voc_1h"),
        },
        "events": events,
        "model_context": {
            "model": state.model_name,
            "prediction_horizon": "15 minutes",
            "warning_basis": "max(current_pm25, predicted_pm25)",
            "features": state.features,
        },
        "llm_consultation_prompt": (
            "You are an aquaculture hatchery air-quality consultant. Analyze this PM2.5 report and provide: "
            "likely causes, operational risk, immediate actions, prevention recommendations, and sensor reliability notes."
        ),
    }


def compact_llm_report(report):
    return {
        "facility": report["facility"],
        "generated_at": report["generated_at"],
        "time_range": report["time_range"],
        "aggregation": report["aggregation"],
        "current_status": report["current_status"],
        "sensor_snapshot": report["sensor_snapshot"],
        "recent_statistics": report["recent_statistics"],
        "events": report["events"][-5:],
        "model_context": {
            "prediction_horizon": report["model_context"]["prediction_horizon"],
            "warning_basis": report["model_context"]["warning_basis"],
        },
    }


def consultation_prompt(report):
    return (
        "You are an aquaculture hatchery air-quality consultant.\n"
        "Analyze the PM2.5 monitoring report below for the hatchery responsible person.\n\n"
        "Return a brief operator-facing consultation in markdown. Keep it under 180 words.\n"
        "Use exactly these sections:\n"
        "## Situation\n"
        "## Risk\n"
        "## Recommended Actions\n"
        "## Notes\n\n"
        "Be practical and specific. Do not mention any LLM, API provider, or model name. "
        "Do not invent facts outside the data. If data is insufficient, say so briefly.\n\n"
        f"PM2.5 report JSON:\n{json.dumps(compact_llm_report(report), indent=2)}"
    )


def call_nvidia_llm(prompt):
    load_env_file()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY in telegram.env before using Analyze.")

    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API error {exc.code}: {detail}") from exc

    return {
        "model": model,
        "content": payload["choices"][0]["message"]["content"],
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




class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = REACT_DASHBOARD_PATH.read_bytes()
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
                query = parse_qs(parsed.query)
                range_hours = int(query.get("range_hours", ["24"])[0])
                interval = query.get("interval", ["15min"])[0]
                if range_hours not in [1, 6, 12, 24]:
                    raise ValueError("range_hours must be one of 1, 6, 12, 24")
                if interval not in ["5min", "15min", "30min"]:
                    raise ValueError("interval must be one of 5min, 15min, 30min")
                json_response(self, history_payload(STATE.live_source, range_hours=range_hours, interval=interval))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/summary":
            try:
                json_response(self, sensor_summary_payload(STATE.live_source))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/report":
            try:
                query = parse_qs(parsed.query)
                range_hours = int(query.get("range_hours", ["24"])[0])
                interval = query.get("interval", ["15min"])[0]
                json_response(self, report_payload(STATE, range_hours=range_hours, interval=interval))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/analyze":
            try:
                query = parse_qs(parsed.query)
                range_hours = int(query.get("range_hours", ["24"])[0])
                interval = query.get("interval", ["15min"])[0]
                report = report_payload(STATE, range_hours=range_hours, interval=interval)
                result = call_nvidia_llm(consultation_prompt(report))
                json_response(self, {"analysis": result["content"], "model": result["model"], "report": compact_llm_report(report)})
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

