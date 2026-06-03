import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import load_env_file

TELEGRAM_API = "https://api.telegram.org"


class TelegramConfigError(RuntimeError):
    pass


def get_telegram_config():
    load_env_file()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramConfigError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before sending Telegram alarms."
        )
    return token, chat_id


def send_telegram_message(text, token=None, chat_id=None, timeout=15):
    if token is None or chat_id is None:
        token, chat_id = get_telegram_config()

    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    body = urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")

    data = json.loads(payload)
    if not data.get("ok"):
        raise RuntimeError(f"Telegram rejected the alarm message: {data}")
    return data
