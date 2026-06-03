import argparse
import csv
import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "live_sensor.csv"
HEADER = ["timestamp(UTC+1)", "score", "temp", "humid", "co2", "voc", "pm25", "pm10"]
LOCAL_TZ = timezone(timedelta(hours=1))


SCENARIOS = {
    "normal": {
        "pm25_base": 7,
        "pm25_event": 0,
        "co2_base": 430,
        "voc_base": 35,
    },
    "moderate": {
        "pm25_base": 18,
        "pm25_event": 8,
        "co2_base": 650,
        "voc_base": 90,
    },
    "high": {
        "pm25_base": 38,
        "pm25_event": 20,
        "co2_base": 900,
        "voc_base": 160,
    },
    "unhealthy": {
        "pm25_base": 58,
        "pm25_event": 35,
        "co2_base": 1200,
        "voc_base": 260,
    },
}


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def read_last_timestamp(path):
    if not path.exists() or path.stat().st_size == 0:
        return None

    last_row = None
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last_row = row

    if not last_row:
        return None

    return datetime.fromisoformat(last_row["timestamp(UTC+1)"])


def ensure_header(path, reset=False):
    if reset and path.exists():
        path.unlink()

    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)


def make_reading(timestamp, step, scenario, seed_noise=True):
    cfg = SCENARIOS[scenario]
    hour_angle = 2 * math.pi * (timestamp.hour + timestamp.minute / 60) / 24
    day_temp = 18 + 5 * math.sin(hour_angle - math.pi / 2)
    humidity = 72 - 18 * math.sin(hour_angle - math.pi / 2)

    event_wave = 0
    if cfg["pm25_event"]:
        event_wave = cfg["pm25_event"] * max(0, math.sin(step / 4))

    noise = random.uniform(-1.5, 1.5) if seed_noise else 0
    pm25 = clamp(cfg["pm25_base"] + event_wave + noise, 0, 250)
    pm10 = clamp(pm25 * random.uniform(1.1, 1.45), 0, 400)
    co2 = clamp(cfg["co2_base"] + event_wave * 12 + random.uniform(-30, 30), 400, 5000)
    voc = clamp(cfg["voc_base"] + event_wave * 4 + random.uniform(-12, 12), 0, 1200)
    temp = clamp(day_temp + random.uniform(-0.4, 0.4), -10, 45)
    humid = clamp(humidity + random.uniform(-2.0, 2.0), 20, 100)

    score = clamp(100 - pm25 * 0.9 - max(co2 - 800, 0) * 0.015 - voc * 0.03, 0, 100)

    return {
        "timestamp(UTC+1)": timestamp.isoformat(sep=" ", timespec="seconds"),
        "score": round(score, 2),
        "temp": round(temp, 2),
        "humid": round(humid, 2),
        "co2": round(co2, 2),
        "voc": round(voc, 2),
        "pm25": round(pm25, 2),
        "pm10": round(pm10, 2),
    }


def append_reading(path, reading):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(reading)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate AQUAIR hatchery sensor readings into a live CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Live CSV to append readings to.")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="normal")
    parser.add_argument("--rows", type=int, default=13, help="Number of readings to append.")
    parser.add_argument("--interval-minutes", type=int, default=5, help="Simulated minutes between rows.")
    parser.add_argument("--sleep-seconds", type=float, default=0, help="Real seconds to wait between appended rows.")
    parser.add_argument("--reset", action="store_true", help="Clear the output CSV before simulating.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_header(args.output, reset=args.reset)

    last_timestamp = read_last_timestamp(args.output)
    if last_timestamp is None:
        timestamp = datetime.now(tz=LOCAL_TZ).replace(second=0, microsecond=0)
    else:
        timestamp = last_timestamp + timedelta(minutes=args.interval_minutes)

    for step in range(args.rows):
        reading = make_reading(timestamp, step, args.scenario)
        append_reading(args.output, reading)
        print(
            f"Appended {reading['timestamp(UTC+1)']} | "
            f"PM2.5={reading['pm25']} | CO2={reading['co2']} | scenario={args.scenario}"
        )
        timestamp += timedelta(minutes=args.interval_minutes)
        if args.sleep_seconds > 0 and step < args.rows - 1:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
