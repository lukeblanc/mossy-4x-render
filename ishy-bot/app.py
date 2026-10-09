"""Flask application that exposes Mushy Merch automation hooks."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from flask import Flask, jsonify
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ishy-bot")

client = OpenAI()

CHECK_INTERVAL_MIN = int(os.getenv("CHECK_INTERVAL_MIN", "30"))
OPENAI_RESPONSE_MODEL = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4o-mini")
MUSHY_MERCH_WEBHOOK_URL = os.getenv("MUSHY_MERCH_WEBHOOK_URL")
STORE_STATUS_URL = os.getenv("STORE_STATUS_URL")

scheduler = BackgroundScheduler()


def fetch_store_status() -> Dict[str, Any]:
    """Fetch the latest store status payload required by the agent."""
    if not STORE_STATUS_URL:
        logger.debug("STORE_STATUS_URL not configured; returning default status payload")
        return {"inventory": [], "notes": "No status endpoint configured."}

    try:
        response = requests.get(STORE_STATUS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.exception("Failed to fetch store status: %s", exc)
        return {"inventory": [], "notes": f"Fetch error: {exc}"}


def build_agent_instructions(status: Dict[str, Any]) -> str:
    """Build the instruction payload passed into the agent."""
    notes = status.get("notes", "No additional notes available.")
    inventory = status.get("inventory", [])
    inventory_lines = "\n".join(
        f"- {item.get('name', 'Unnamed item')} (qty: {item.get('quantity', 'unknown')})"
        for item in inventory
    ) or "- No inventory data"

    return (
        "You are Ishy, an automation bot that manages Mushy Merch tasks. "
        "Review the inventory list and notes, then provide concise next actions.\n"
        f"Notes: {notes}\n"
        f"Inventory:\n{inventory_lines}\n"
        "Respond with a bullet list of actions."
    )


def invoke_agent(status: Dict[str, Any]) -> str:
    """Invoke the OpenAI model to determine next actions."""
    prompt = build_agent_instructions(status)

    logger.debug("Sending prompt to OpenAI model %s", OPENAI_RESPONSE_MODEL)

    try:
        response = client.responses.create(
            model=OPENAI_RESPONSE_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "You help operate the Mushy Merch storefront efficiently.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:  # broad catch to ensure reliability of worker
        logger.exception("OpenAI invocation failed: %s", exc)
        return f"OpenAI invocation failed: {exc}"

    try:
        message = response.output_text.strip()
    except AttributeError:
        logger.warning("Unexpected response payload from OpenAI: %s", response)
        message = "Unable to parse response payload."

    return message


def dispatch_actions(message: str) -> None:
    """Send the agent output to the configured webhook."""
    if not MUSHY_MERCH_WEBHOOK_URL:
        logger.info("Webhook URL not configured; skipping dispatch. Message: %s", message)
        return

    payload = {"text": message, "timestamp": datetime.utcnow().isoformat()}

    try:
        response = requests.post(
            MUSHY_MERCH_WEBHOOK_URL, json=payload, timeout=10
        )
        response.raise_for_status()
        logger.info("Successfully dispatched Mushy Merch instructions")
    except requests.RequestException as exc:
        logger.exception("Failed to dispatch actions: %s", exc)


def run_agent_cycle() -> Dict[str, Any]:
    """Run a full cycle: fetch status, invoke the agent, and dispatch actions."""
    logger.info("Starting Mushy Merch agent cycle")
    status = fetch_store_status()
    message = invoke_agent(status)
    dispatch_actions(message)

    return {"status": status, "message": message}


def schedule_agent() -> None:
    """Start the scheduler if not already running."""
    if scheduler.running:
        logger.debug("Scheduler already running; skipping start")
        return

    trigger = IntervalTrigger(minutes=CHECK_INTERVAL_MIN)
    scheduler.add_job(run_agent_cycle, trigger, name="mushy-merch-cycle")
    scheduler.start()
    logger.info(
        "Scheduled Mushy Merch agent to run every %s minutes", CHECK_INTERVAL_MIN
    )


def create_app() -> Flask:
    """Factory to create the Flask application."""
    flask_app = Flask(__name__)

    @flask_app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

    @flask_app.route("/run", methods=["POST"])
    def run() -> Any:
        result = run_agent_cycle()
        return jsonify(result)

    schedule_agent()
    return flask_app


app = create_app()


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_agent_cycle()
    else:
        port = int(os.getenv("PORT", "8000"))
        app.run(host="0.0.0.0", port=port)
