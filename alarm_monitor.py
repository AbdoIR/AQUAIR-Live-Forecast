import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from config import load_env_file
import model_interface as warning_model
from telegram_alarm import TelegramConfigError, send_telegram_message


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "live_sensor.csv"
DEFAULT_STATE_PATH = BASE_DIR / "models" / "alarm_state.json"
DANGEROUS_PM25 = 35.0
LEVEL_RANK = {
    "Normal / Good": 0,
    "Moderate": 1,
    "High Pollution": 2,
    "Unhealthy": 3,
}


def load_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(path, state):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def format_alarm_message(result, hatchery_name):
    predicted = result["prediction"]
    risk_pm25 = result.get("risk_pm25", predicted)
    risk_source = result.get("risk_source", "15-minute forecast")
    level = result["level"]
    trend = result["trend"]

    return "\n".join(
        [
            "AQUAIR PM2.5 ALARM" + (" [SIMULATION]" if result.get("simulation") else ""),
            f"Facility: {hatchery_name}",
            f"Level: {level['name']}",
            f"PM2.5 used for warning: {risk_pm25:.2f} ug/m3 ({risk_source})",
            f"Predicted PM2.5 in 15 min: {predicted:.2f} ug/m3",
            f"Current PM2.5: {trend['current_pm25']:.2f} ug/m3",
            f"Trend: {trend['direction']} ({trend['delta_next_15m']:+.2f} ug/m3 expected)",
            f"Action: {level['action']}",
            f"Timestamp: {result['latest_timestamp']}",
            f"Model: {result['model_name']}",
        ]
    )


def apply_simulated_prediction(result, simulated_prediction):
    if simulated_prediction is None or not result.get("ready", False):
        return result

    result = dict(result)
    trend = dict(result["trend"])
    prediction = float(simulated_prediction)
    trend["delta_next_15m"] = prediction - trend["current_pm25"]

    if trend["delta_next_15m"] >= 3:
        trend["direction"] = "rising"
    elif trend["delta_next_15m"] <= -3:
        trend["direction"] = "falling"
    else:
        trend["direction"] = "stable"

    result["prediction"] = prediction
    result["risk_pm25"] = max(prediction, trend["current_pm25"])
    result["risk_source"] = "current reading" if trend["current_pm25"] >= prediction else "15-minute forecast"
    result["level"] = warning_model.warning_level(result["risk_pm25"])
    result["trend"] = trend
    result["simulation"] = True
    return result


def should_send_alarm(result, state, min_pm25, cooldown_minutes):
    risk_pm25 = result.get("risk_pm25", result["prediction"])
    level_name = result["level"]["name"]

    if risk_pm25 < min_pm25:
        return False, "PM2.5 risk value is below the alarm threshold."

    now = time.time()
    last_sent_at = float(state.get("last_sent_at", 0))
    last_level = state.get("last_level")
    cooldown_seconds = cooldown_minutes * 60

    level_got_worse = LEVEL_RANK.get(level_name, 0) > LEVEL_RANK.get(last_level, -1)
    cooldown_elapsed = now - last_sent_at >= cooldown_seconds
    if cooldown_elapsed or level_got_worse:
        return True, "Alarm threshold reached."

    remaining = int((cooldown_seconds - (now - last_sent_at)) / 60)
    return False, f"Alarm already sent recently; cooldown has about {remaining} minutes remaining."


def update_last_seen(state, result):
    state["last_seen_timestamp"] = result.get("latest_timestamp")
    state["last_seen_iso"] = datetime.now().isoformat(timespec="seconds")
    return state


def check_once(args):
    app_state = warning_model.load_model_state(args.source, args.history_size)
    result = warning_model.live_warning_payload(app_state)
    result = apply_simulated_prediction(result, args.simulate_prediction)
    state = load_state(args.state_path)

    if not result.get("ready", False):
        print(result["message"])
        if result.get("available_rows") == 0 and result.get("expected_header"):
            print("Expected live CSV header:")
            print(result["expected_header"])
        return result

    latest_timestamp = result["latest_timestamp"]
    if state.get("last_seen_timestamp") == latest_timestamp and not args.simulate_prediction:
        print(f"No new live row. Latest timestamp already processed: {latest_timestamp}")
        return result

    should_send, reason = should_send_alarm(result, state, args.min_pm25, args.cooldown_minutes)
    message = format_alarm_message(result, args.hatchery_name)
    state = update_last_seen(state, result)

    if should_send:
        if args.dry_run:
            print("[DRY RUN] Telegram message would be sent:")
            print(message)
            print("[DRY RUN] Cooldown state was not updated.")
            return result

        send_telegram_message(message)
        print("Telegram alarm sent.")
        state.update(
            {
                "last_sent_at": time.time(),
                "last_sent_iso": datetime.now().isoformat(timespec="seconds"),
                "last_level": result["level"]["name"],
                "last_prediction": result["prediction"],
                "last_risk_pm25": result.get("risk_pm25", result["prediction"]),
            }
        )
    else:
        print(f"No Telegram alarm sent: {reason}")
        print(
            f"Level={result['level']['name']}; "
            f"risk PM2.5={result.get('risk_pm25', result['prediction']):.2f} ug/m3; "
            f"predicted PM2.5={result['prediction']:.2f} ug/m3"
        )

    save_state(args.state_path, state)
    return result


def parse_args():
    load_env_file()
    parser = argparse.ArgumentParser(description="Send Telegram alarms from live PM2.5 CSV history.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("LIVE_SENSOR_CSV", DEFAULT_SOURCE)),
        help="Live CSV containing appended AQUAIR readings.",
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH, help="Cooldown and last-seen state JSON path.")
    parser.add_argument("--history-size", type=int, default=warning_model.DEFAULT_HISTORY_SIZE)
    parser.add_argument("--min-pm25", type=float, default=float(os.getenv("ALARM_MIN_PM25", DANGEROUS_PM25)))
    parser.add_argument("--cooldown-minutes", type=int, default=30, help="Minimum time between repeated alarms.")
    parser.add_argument("--hatchery-name", default=os.getenv("HATCHERY_NAME", "Azrou hatchery"))
    parser.add_argument("--interval-seconds", type=int, default=0, help="Repeat forever every N seconds when greater than 0.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Telegram message without sending it.")
    parser.add_argument(
        "--simulate-prediction",
        type=float,
        default=None,
        help="Override predicted PM2.5 for alarm-message testing, for example 60 for Unhealthy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dry_run:
        try:
            from telegram_alarm import get_telegram_config

            get_telegram_config()
        except TelegramConfigError as exc:
            raise SystemExit(str(exc)) from exc

    while True:
        check_once(args)
        if args.interval_seconds <= 0:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
