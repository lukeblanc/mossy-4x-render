import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

import requests

REPO = "lukeblanc/mossy-4x-render"
GITHUB_API_BASE = "https://api.github.com"
RENDER_API_BASE = "https://api.render.com/v1"
SERVICES_ENDPOINT = f"{RENDER_API_BASE}/services"
SERVICE_DETAIL_ENDPOINT = f"{RENDER_API_BASE}/services/{{service_id}}"
SERVICE_ENV_VARS_ENDPOINT = f"{RENDER_API_BASE}/services/{{service_id}}/env-vars"
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 30 * 60
HEALTH_TIMEOUT_SECONDS = 10 * 60
EXPECTED_HEALTH_RESPONSE = {"service": "MCP"}


def mask_secret(value: str) -> str:
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


def create_render_session(render_api_token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {render_api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    return session


def print_http_error(prefix: str, response: requests.Response, replacements: Dict[str, str]) -> None:
    try:
        payload = response.json()
        detail = json.dumps(payload, indent=2)
    except ValueError:
        detail = response.text
    message = f"{prefix}: HTTP {response.status_code} - {detail}"
    print(sanitize(message, replacements))


def fetch_services(session: requests.Session) -> Iterable[dict]:
    response = session.get(SERVICES_ENDPOINT, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "services" in data and isinstance(data["services"], list):
        return data["services"]
    return []


def extract_service_id(service_payload: dict) -> Optional[str]:
    if not isinstance(service_payload, dict):
        return None
    if "id" in service_payload and isinstance(service_payload["id"], str):
        return service_payload["id"]
    if "service" in service_payload and isinstance(service_payload["service"], dict):
        service_dict = service_payload["service"]
        if "id" in service_dict and isinstance(service_dict["id"], str):
            return service_dict["id"]
    return None


def extract_service_status(service_payload: dict) -> str:
    if not isinstance(service_payload, dict):
        return ""
    if "serviceStatus" in service_payload:
        return str(service_payload.get("serviceStatus") or "")
    if "status" in service_payload:
        return str(service_payload.get("status") or "")
    if "service" in service_payload and isinstance(service_payload["service"], dict):
        service_dict = service_payload["service"]
        status = service_dict.get("serviceStatus") or service_dict.get("status")
        if isinstance(status, str):
            return status
    return ""


def find_service_by_name(session: requests.Session, name: str) -> Optional[dict]:
    for entry in fetch_services(session):
        if not isinstance(entry, dict):
            continue
        entry_name = entry.get("name")
        if not isinstance(entry_name, str) and "service" in entry and isinstance(entry["service"], dict):
            entry_name = entry["service"].get("name")
        if entry_name == name:
            return entry
    return None


def fetch_service_detail(session: requests.Session, service_id: str) -> dict:
    response = session.get(
        SERVICE_DETAIL_ENDPOINT.format(service_id=service_id),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    return {}


def create_service(
    session: requests.Session,
    payload: Dict[str, object],
    replacements: Dict[str, str],
) -> Optional[str]:
    response = session.post(SERVICES_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 409:
        existing = find_service_by_name(session, str(payload.get("name")))
        return extract_service_id(existing) if existing else None
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print_http_error("Render service creation failed", response, replacements)
        return None
    data = response.json()
    return extract_service_id(data)


def fetch_env_map(
    session: requests.Session,
    service_id: str,
    replacements: Dict[str, str],
) -> Dict[str, str]:
    response = session.get(
        SERVICE_ENV_VARS_ENDPOINT.format(service_id=service_id),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print_http_error("Failed to fetch Render environment variables", response, replacements)
        return {}
    data = response.json()
    env_map: Dict[str, str] = {}
    items: Iterable = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("envVars"), list):
            items = data["envVars"]
        else:
            items = data.values()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and isinstance(value, str):
            env_map[key] = value
    return env_map


def put_env_vars(
    session: requests.Session,
    service_id: str,
    env_vars: List[Dict[str, str]],
    replacements: Dict[str, str],
) -> bool:
    response = session.put(
        SERVICE_ENV_VARS_ENDPOINT.format(service_id=service_id),
        json={"envVars": env_vars},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print_http_error("Failed to update Render environment variables", response, replacements)
        return False
    return True


def wait_for_services(
    session: requests.Session,
    services: Dict[str, str],
    replacements: Dict[str, str],
) -> Dict[str, str]:
    start = time.monotonic()
    last_statuses: Dict[str, str] = {name: "" for name in services}
    while True:
        all_live = True
        for name, service_id in services.items():
            try:
                detail = fetch_service_detail(session, service_id)
            except requests.RequestException as exc:
                print(sanitize(f"Error fetching status for {name}: {exc}", replacements))
                last_statuses[name] = "error"
                all_live = False
                continue
            status = extract_service_status(detail)
            last_statuses[name] = status or "unknown"
            if (status or "").lower() != "live":
                all_live = False
        if all_live:
            return last_statuses
        if time.monotonic() - start > WAIT_TIMEOUT_SECONDS:
            status_line = ", ".join(f"{n}={s}" for n, s in last_statuses.items())
            print(
                sanitize(
                    f"Timed out waiting for services to become live: {status_line}",
                    replacements,
                )
            )
            return last_statuses
        status_line = ", ".join(f"{n}={s or 'unknown'}" for n, s in last_statuses.items())
        print(sanitize(f"Waiting for services: {status_line}", replacements))
        time.sleep(POLL_INTERVAL_SECONDS)


def trigger_render_redeploy(url: str, replacements: Dict[str, str]) -> bool:
    try:
        response = requests.post(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return True
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            print_http_error("Render redeploy failed", response, replacements)
        else:
            print(sanitize(f"Render redeploy failed: {exc}", replacements))
    except requests.RequestException as exc:
        print(sanitize(f"Render redeploy request error: {exc}", replacements))
    return False


def dispatch_workflow(
    gh_pat: str,
    workflow_file: str,
    replacements: Dict[str, str],
) -> bool:
    url = f"{GITHUB_API_BASE}/repos/{REPO}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {gh_pat}",
        "Accept": "application/vnd.github+json",
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json={"ref": "main"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            print_http_error("GitHub workflow dispatch failed", response, replacements)
        else:
            print(sanitize(f"GitHub workflow dispatch failed: {exc}", replacements))
    except requests.RequestException as exc:
        print(sanitize(f"GitHub workflow request error: {exc}", replacements))
    return False


def extract_service_url(service_detail: dict) -> Optional[str]:
    if not isinstance(service_detail, dict):
        return None
    if "service" in service_detail and isinstance(service_detail["service"], dict):
        inner = service_detail["service"]
        for key in ("serviceDetails", "staticSite", "service" ):
            value = inner.get(key)
            if isinstance(value, dict):
                url = value.get("url") or value.get("dashboardUrl")
                if isinstance(url, str):
                    return url
        url = inner.get("url")
        if isinstance(url, str):
            return url
        domain = inner.get("domain") or inner.get("defaultDomain")
        if isinstance(domain, dict):
            for candidate in ("name", "url", "serviceUrl"):
                url = domain.get(candidate)
                if isinstance(url, str):
                    return url
    if "url" in service_detail and isinstance(service_detail["url"], str):
        return service_detail["url"]
    return None


def poll_health_endpoint(url: str, replacements: Dict[str, str]) -> bool:
    if not url:
        print("Unable to determine MCP web URL for health check")
        return False
    if not url.startswith("http"):
        url = f"https://{url}" if not url.startswith("//") else f"https:{url}"
    if not url.endswith("/health"):
        health_url = url.rstrip("/") + "/health"
    else:
        health_url = url

    start = time.monotonic()
    while True:
        try:
            response = requests.get(health_url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("service") == EXPECTED_HEALTH_RESPONSE["service"]:
                print(f"Health check succeeded at {health_url}")
                return True
            print(
                sanitize(
                    f"Health check returned unexpected payload: {payload}",
                    replacements,
                )
            )
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None:
                print_http_error("Health check failed", response, replacements)
            else:
                print(sanitize(f"Health check failed: {exc}", replacements))
        except (requests.RequestException, ValueError) as exc:
            print(sanitize(f"Health check error: {exc}", replacements))
        if time.monotonic() - start > HEALTH_TIMEOUT_SECONDS:
            print("Health check timed out")
            return False
        time.sleep(POLL_INTERVAL_SECONDS)


def build_env_vars(
    mcp_api_key: str,
    gh_pat: str,
    hook_url: str,
    oanda_env: Dict[str, str],
) -> List[Dict[str, str]]:
    env_vars = [
        {"key": "MCP_API_KEY", "value": mcp_api_key},
        {"key": "GH_PAT", "value": gh_pat},
        {"key": "RENDER_DEPLOY_HOOK_URL", "value": hook_url},
    ]
    for key in ("OANDA_API_KEY", "OANDA_ACCOUNT_ID"):
        if key in oanda_env:
            env_vars.append({"key": key, "value": oanda_env[key]})
    return env_vars


def format_status(status: str) -> str:
    if (status or "").lower() == "live":
        return "✅ Live"
    if not status:
        return "⚠️ Unknown"
    return f"⚠️ {status}"


def print_summary_table(statuses: Dict[str, str]) -> None:
    print("| Component | Status |")
    print("|-----------|--------|")
    for component in ("mossy-4x", "mcp-web", "mcp-optimiser", "auto-mossy.yml"):
        status = statuses.get(component, "⚠️ Unknown")
        print(f"| {component} | {status} |")


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
        render_api_token: mask_secret(render_api_token),
        mcp_api_key: mask_secret(mcp_api_key),
        gh_pat: mask_secret(gh_pat),
        render_deploy_hook_url: mask_secret(render_deploy_hook_url),
    }

    session = create_render_session(render_api_token)

    statuses: Dict[str, str] = {
        "mossy-4x": "⚠️ Unknown",
        "mcp-web": "⚠️ Pending",
        "mcp-optimiser": "⚠️ Pending",
        "auto-mossy.yml": "⚠️ Not triggered",
    }

    oanda_env: Dict[str, str] = {}
    mossy_service_id: Optional[str] = None

    try:
        mossy_service = find_service_by_name(session, "mossy-4x")
        if mossy_service:
            mossy_service_id = extract_service_id(mossy_service)
            statuses["mossy-4x"] = format_status(extract_service_status(mossy_service))
            if mossy_service_id:
                oanda_env = fetch_env_map(session, mossy_service_id, replacements)
        else:
            statuses["mossy-4x"] = "⚠️ Not found"
    except requests.RequestException as exc:
        print(sanitize(f"Failed to retrieve mossy-4x service: {exc}", replacements))
        statuses["mossy-4x"] = "⚠️ Error"

    env_vars = build_env_vars(mcp_api_key, gh_pat, render_deploy_hook_url, oanda_env)

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
            "rootDir": "mcp",
            "dockerfilePath": "mcp/Dockerfile",
        },
        "envVars": env_vars,
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
            "rootDir": "mcp",
            "dockerfilePath": "mcp/Dockerfile",
        },
        "envVars": env_vars,
        "serviceDetails": {
            "startCommand": "python optimiser.py && python patcher.py",
            "schedule": "0 1 * * *",
        },
        "autoDeploy": True,
    }

    service_ids: Dict[str, str] = {}

    for name, payload in (("mcp-web", web_payload), ("mcp-optimiser", worker_payload)):
        try:
            service_id = create_service(session, payload, replacements)
            if not service_id:
                print(sanitize(f"Unable to ensure {name} service", replacements))
                statuses[name] = "❌ Creation failed"
                continue
            service_ids[name] = service_id
            if put_env_vars(session, service_id, env_vars, replacements):
                print(sanitize(f"Updated environment variables for {name}", replacements))
            else:
                statuses[name] = "⚠️ Env update failed"
        except requests.RequestException as exc:
            print(sanitize(f"Error ensuring {name}: {exc}", replacements))
            statuses[name] = "❌ Error"

    if service_ids:
        live_statuses = wait_for_services(session, service_ids, replacements)
        for name, status in live_statuses.items():
            formatted = format_status(status)
            if statuses.get(name, "").startswith("❌"):
                continue
            statuses[name] = formatted
    else:
        print("No services created; skipping wait for live status")

    if not trigger_render_redeploy(render_deploy_hook_url, replacements):
        print("Render redeploy did not complete successfully")

    detail_for_health: Optional[dict] = None
    if "mcp-web" in service_ids:
        try:
            detail_for_health = fetch_service_detail(session, service_ids["mcp-web"])
        except requests.RequestException as exc:
            print(sanitize(f"Failed to fetch mcp-web detail: {exc}", replacements))

    web_url = extract_service_url(detail_for_health or {}) if detail_for_health else None
    if "mcp-web" in service_ids and statuses.get("mcp-web", "").startswith("✅"):
        if not poll_health_endpoint(web_url or "", replacements):
            statuses["mcp-web"] = "⚠️ Health check failed"

    if dispatch_workflow(gh_pat, "finish-render-setup.yml", replacements):
        print("Dispatched finish-render-setup workflow")
    if dispatch_workflow(gh_pat, "auto-mossy.yml", replacements):
        statuses["auto-mossy.yml"] = "✅ Last run dispatched"
    else:
        statuses["auto-mossy.yml"] = "⚠️ Dispatch failed"

    print_summary_table(statuses)
    print("✅ Mossy 4X system fully operational.")


if __name__ == "__main__":
    main()
