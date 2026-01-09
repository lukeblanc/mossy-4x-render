import os
import requests


def send_bot_status(status: str):
    url = os.getenv("DASHBOARD_URL")
    key = os.getenv("INGEST_API_KEY")

    if not url or not key:
        return

    try:
        requests.post(
            f"{url}/api/ingest",
            json={
                "api_key": key,
                "type": "bot_status",
                "data": {"status": status},
            },
            timeout=3,
        )
    except Exception:
        pass


def send_snapshot(user_id: str, balance: float):
    url = os.getenv("DASHBOARD_URL")
    key = os.getenv("INGEST_API_KEY")

    if not url or not key:
        return

    try:
        requests.post(
            f"{url}/api/ingest",
            json={
                "api_key": key,
                "type": "snapshot",
                "data": {
                    "balance": balance,
                },
            },
            timeout=3,
        )
    except Exception:
        pass

