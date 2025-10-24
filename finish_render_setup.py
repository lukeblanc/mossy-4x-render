import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

import requests

RENDER_API_BASE = "https://api.render.com/v1"
SERVICES_ENDPOINT = f"{RENDER_API_BASE}/services"
SERVICE_DETAIL_ENDPOINT = f"{RENDER_API_BASE}/services/{{service_id}}"
GITHUB_DISPATCH_URL = (
    "https://api.github.com/repos/lukeblanc/mossy-4x-render/actions/workflows/auto-mossy.yml/dispatches"
)
POLL_INTERVAL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 30 * 60


def mask_token(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:6]}..."


def sanitize(message: str, replacements: Dict[str, str]) -> str:
    sanitized = message
    for secret, masked in replacements.items():
        if secret:
            sanitized = sanitized.replace(secret, masked)
    return sanitized


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def create_service(
    session: requests.Session,
    payload: Dict[str, object],
    replacements: Dict[str, str],
) -> str:
    response = session.post(SERVICES_ENDPOINT, json=payload, timeout=30)
    if response.status_code == 409:
        existing_id = find_service_id(session, payload.get("name"))
        if existing_id:
            return existing_id
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print_http_error("Render service creation failed", response, replacements)
        raise exc
    data = response.json()
    if isinstance(data, dict):
        service_id = data.get("id") or (
            isinstance(data.get("service"), dict) and data["service"].get("id")
        )
        if isinstance(service_id, str):
            return service_id
    raise RuntimeError(
        sanitize(
            f"Render service creation returned unexpected payload: {response.text}",
            replacements,
        )
    )


def find_service_id(session: requests.Session, name: Optional[object]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    response = session.get(SERVICES_ENDPOINT, timeout=30)
    if response.status_code >= 400:
        return None
    try:
        services: Iterable[object] = response.json()
    except ValueError:
        return None
    for entry in services:
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("name")
        if entry_name == name:
            service_id = entry.get("id")
            if isinstance(service_id, str):
                return service_id
    return None


def fetch_service_status(session: requests.Session, service_id: str) -> str:
    response = session.get(
        SERVICE_DETAIL_ENDPOINT.format(service_id=service_id), timeout=30
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "service" in data and isinstance(data["service"], dict):
            return str(data["service"].get("serviceStatus") or data["service"].get("status") or "")
        status = data.get("serviceStatus") or data.get("status")
        if isinstance(status, str):
            return status
    return ""


def wait_for_services(
    session: requests.Session,
    services: Dict[str, str],
    replacements: Dict[str, str],
) -> None:
    start = time.monotonic()
    while True:
        statuses: Dict[str, str] = {}
        for name, service_id in services.items():
            status = fetch_service_status(session, service_id)
            statuses[name] = status
        if all(status.lower() == "live" for status in statuses.values() if status):
            return
        if time.monotonic() - start > WAIT_TIMEOUT_SECONDS:
            status_line = ", ".join(f"{name}={status or 'unknown'}" for name, status in statuses.items())
            raise TimeoutError(
                sanitize(
                    f"Timed out waiting for services to become live: {status_line}",
                    replacements,
                )
            )
        status_line = ", ".join(f"{name}={status or 'unknown'}" for name, status in statuses.items())
        print(sanitize(f"Waiting for services: {status_line}", replacements))
        time.sleep(POLL_INTERVAL_SECONDS)


def dispatch_workflow(gh_pat: str, replacements: Dict[str, str]) -> None:
    response = requests.post(
        GITHUB_DISPATCH_URL,
        headers={
            "Authorization": f"Bearer {gh_pat}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": "main"},
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print_http_error("GitHub workflow dispatch failed", response, replacements)
        raise exc


def print_http_error(prefix: str, response: requests.Response, replacements: Dict[str, str]) -> None:
    try:
        detail_obj = response.json()
        detail = json.dumps(detail_obj, indent=2)
    except ValueError:
        detail = response.text
    message = f"{prefix}: HTTP {response.status_code} - {detail}"
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

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {render_api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    shared_env_vars: List[Dict[str, str]] = [
        {"key": "MCP_API_KEY", "value": mcp_api_key},
        {"key": "GH_PAT", "value": gh_pat},
        {"key": "RENDER_DEPLOY_HOOK_URL", "value": render_deploy_hook_url},
    ]

    web_payload = {
        "type": "web_service",
        "name": "mcp-web",
        "plan": "free",
        "runtime": "docker",
        "repo": {
            "type": "github",
            "owner": "lukeblanc",
            "name": "mossy-4x-render",
            "branch": "main",
            "path": "mcp",
        },
        "envVars": shared_env_vars,
        "autoDeploy": True,
    }

    worker_payload = {
        "type": "background_worker",
        "name": "mcp-optimiser",
        "plan": "starter",
        "runtime": "docker",
        "repo": {
            "type": "github",
            "owner": "lukeblanc",
            "name": "mossy-4x-render",
            "branch": "main",
            "path": "mcp",
        },
        "envVars": shared_env_vars,
        "serviceDetails": {
            "startCommand": "python optimiser.py && python patcher.py",
            "schedule": "0 1 * * *",
        },
        "autoDeploy": True,
    }

    try:
        web_id = create_service(session, web_payload, replacements)
        worker_id = create_service(session, worker_payload, replacements)
    except requests.RequestException:
        sys.exit(1)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    services = {"mcp-web": web_id, "mcp-optimiser": worker_id}

    try:
        wait_for_services(session, services, replacements)
    except TimeoutError as exc:
        print(exc)
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
        dispatch_workflow(gh_pat, replacements)
    except requests.RequestException:
        sys.exit(1)

    print("✅ Render services live")
    print("✅ GitHub workflow dispatched")


if __name__ == "__main__":
    main()
