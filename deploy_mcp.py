import json
import os
import sys
import time
from typing import Dict, Iterable, Tuple

import requests

RENDER_APPLY_URL = "https://api.render.com/v1/blueprints/apply"
RENDER_SERVICE_URL = "https://api.render.com/v1/services/{service_id}"
GITHUB_DISPATCH_URL = (
    "https://api.github.com/repos/lukeblanc/mossy-4x-render/actions/workflows/auto-mossy.yml/dispatches"
)
POLL_INTERVAL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 30 * 60


def mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}{'*' * (len(token) - 8)}{token[-4:]}"


def sanitize(text: str, replacements: Dict[str, str]) -> str:
    sanitized = text
    for original, masked in replacements.items():
        if original:
            sanitized = sanitized.replace(original, masked)
    return sanitized


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def apply_blueprint(render_token: str, blueprint_content: str) -> Dict[str, object]:
    headers = {
        "Authorization": f"Bearer {render_token}",
        "Content-Type": "application/x-yaml",
        "Accept": "application/json",
    }
    response = requests.post(
        RENDER_APPLY_URL,
        data=blueprint_content.encode("utf-8"),
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    if not response.text:
        return {}
    try:
        return response.json()
    except json.JSONDecodeError:
        return {}


def extract_services(apply_result: Dict[str, object]) -> Dict[str, str]:
    services: Dict[str, str] = {}

    def ingest_service_entries(entries: Iterable[object]) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            service_data = entry
            if "service" in entry and isinstance(entry["service"], dict):
                service_data = entry["service"]
            name = service_data.get("name") if isinstance(service_data, dict) else None
            service_id = service_data.get("id") if isinstance(service_data, dict) else None
            if not service_id and isinstance(entry, dict):
                service_id = entry.get("id") or entry.get("serviceId") or entry.get("serviceID")
            if name and service_id:
                services[name] = service_id

    if not isinstance(apply_result, dict):
        return services

    if "services" in apply_result and isinstance(apply_result["services"], list):
        ingest_service_entries(apply_result["services"])

    if "resources" in apply_result and isinstance(apply_result["resources"], list):
        ingest_service_entries(apply_result["resources"])

    blueprint = apply_result.get("blueprint")
    if isinstance(blueprint, dict):
        if "services" in blueprint and isinstance(blueprint["services"], list):
            ingest_service_entries(blueprint["services"])
        if "resources" in blueprint and isinstance(blueprint["resources"], list):
            ingest_service_entries(blueprint["resources"])

    if not services:
        service_ids = apply_result.get("serviceIds")
        if isinstance(service_ids, list):
            for index, service_id in enumerate(service_ids, start=1):
                if isinstance(service_id, str):
                    services[f"service_{index}"] = service_id

    return services


def fetch_service_status(render_token: str, service_id: str) -> Tuple[str, str]:
    headers = {
        "Authorization": f"Bearer {render_token}",
        "Accept": "application/json",
    }
    response = requests.get(
        RENDER_SERVICE_URL.format(service_id=service_id),
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    status = None
    name = None
    if isinstance(data, dict):
        if "service" in data and isinstance(data["service"], dict):
            service_block = data["service"]
            status = service_block.get("status")
            name = service_block.get("name")
        else:
            status = data.get("status")
            name = data.get("name")
        if not status and isinstance(data.get("deployment"), dict):
            status = data["deployment"].get("status")
    return (status or "unknown", name or "")


def wait_for_services_live(render_token: str, services: Dict[str, str]) -> None:
    start_time = time.monotonic()
    while True:
        statuses = {}
        for name, service_id in services.items():
            status, reported_name = fetch_service_status(render_token, service_id)
            display_name = reported_name or name
            statuses[display_name] = status
        if all(status.lower() == "live" for status in statuses.values()):
            break
        elapsed = time.monotonic() - start_time
        if elapsed > WAIT_TIMEOUT_SECONDS:
            status_line = ", ".join(f"{svc}={state}" for svc, state in statuses.items())
            raise TimeoutError(f"Timed out waiting for services to become live ({status_line}).")
        status_line = ", ".join(f"{svc}={state}" for svc, state in statuses.items())
        print(f"Waiting for services: {status_line}")
        time.sleep(POLL_INTERVAL_SECONDS)


def dispatch_github_workflow(gh_pat: str) -> None:
    headers = {
        "Authorization": f"Bearer {gh_pat}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.post(
        GITHUB_DISPATCH_URL,
        headers=headers,
        json={"ref": "main"},
        timeout=30,
    )
    response.raise_for_status()


def print_http_error(prefix: str, response: requests.Response, replacements: Dict[str, str]) -> None:
    status = response.status_code
    try:
        detail = json.dumps(response.json(), indent=2)
    except ValueError:
        detail = response.text
    message = f"{prefix}: HTTP {status} - {detail}"
    print(sanitize(message, replacements))


def main() -> None:
    try:
        render_api_token = require_env("RENDER_API_TOKEN")
        mcp_api_key = require_env("MCP_API_KEY")
        gh_pat = require_env("GH_PAT")
        render_deploy_hook_url = require_env("RENDER_DEPLOY_HOOK_URL")
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    replacements = {
        render_api_token: mask_token(render_api_token),
        mcp_api_key: mask_token(mcp_api_key),
        gh_pat: mask_token(gh_pat),
        render_deploy_hook_url: mask_token(render_deploy_hook_url),
    }

    try:
        blueprint_content = open("mcp/render.yaml", "r", encoding="utf-8").read()
    except OSError as exc:
        print(sanitize(f"Failed to read blueprint file: {exc}", replacements))
        sys.exit(1)

    try:
        apply_result = apply_blueprint(render_api_token, blueprint_content)
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            print_http_error("Render blueprint apply failed", response, replacements)
        else:
            print(sanitize(f"Render blueprint apply failed: {exc}", replacements))
        sys.exit(1)
    except requests.RequestException as exc:
        print(sanitize(f"Render blueprint apply request error: {exc}", replacements))
        sys.exit(1)

    services = extract_services(apply_result)
    if not services:
        print("Render blueprint applied but no services were returned in the response.")
        sys.exit(1)

    print("Render blueprint applied; waiting for services to reach 'live' status...")
    try:
        wait_for_services_live(render_api_token, services)
    except TimeoutError as exc:
        print(sanitize(str(exc), replacements))
        sys.exit(1)
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            print_http_error("Failed to fetch Render service status", response, replacements)
        else:
            print(sanitize(f"Failed to fetch Render service status: {exc}", replacements))
        sys.exit(1)
    except requests.RequestException as exc:
        print(sanitize(f"Error while polling Render services: {exc}", replacements))
        sys.exit(1)

    try:
        dispatch_github_workflow(gh_pat)
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            print_http_error("GitHub workflow dispatch failed", response, replacements)
        else:
            print(sanitize(f"GitHub workflow dispatch failed: {exc}", replacements))
        sys.exit(1)
    except requests.RequestException as exc:
        print(sanitize(f"GitHub workflow dispatch error: {exc}", replacements))
        sys.exit(1)

    print("✅ MCP deployed & workflow dispatched")


if __name__ == "__main__":
    main()
