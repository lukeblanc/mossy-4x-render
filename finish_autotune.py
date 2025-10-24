from __future__ import annotations

import base64
import sys
from getpass import getpass
from typing import Tuple

import requests
from nacl import encoding, public

GITHUB_API_BASE = "https://api.github.com"
REPO = "lukeblanc/mossy-4x-render"
SECRET_NAME = "GH_PAT"
REQUEST_TIMEOUT = 30
API_VERSION = "2022-11-28"


def mask_pat(token: str) -> str:
    """Return a masked representation of a GitHub PAT."""

    if token.startswith("ghp_"):
        tail = token[-4:] if len(token) > 4 else ""
        return f"ghp_{'*' * 8}{tail}"
    if len(token) <= 4:
        return "*" * len(token)
    return f"{'*' * 4}{token[-4:]}"


def authorization_headers(token: str) -> dict[str, str]:
    """Headers for authenticated GitHub API requests."""

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def finish_pat_authorization(token: str, otp_code: str) -> None:
    """Send the one-time password to finalize PAT authorization."""

    url = f"{GITHUB_API_BASE}/authorizations"
    payload = {"one_time_password": otp_code}
    response = requests.post(
        url,
        headers=authorization_headers(token),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def get_repo_public_key(token: str) -> Tuple[str, str]:
    """Fetch the repository's public key and key ID."""

    url = f"{GITHUB_API_BASE}/repos/{REPO}/actions/secrets/public-key"
    response = requests.get(
        url,
        headers=authorization_headers(token),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["key"], payload["key_id"]


def encrypt_secret(secret_value: str, public_key: str) -> str:
    """Encrypt a secret using GitHub's provided public key."""

    key = public.PublicKey(public_key, encoding.Base64Encoder())
    sealed_box = public.SealedBox(key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def put_repo_secret(token: str, encrypted_value: str, key_id: str) -> None:
    """Create or update the GH_PAT secret on the repository."""

    url = f"{GITHUB_API_BASE}/repos/{REPO}/actions/secrets/{SECRET_NAME}"
    payload = {"encrypted_value": encrypted_value, "key_id": key_id}
    response = requests.put(
        url,
        headers=authorization_headers(token),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def dispatch_workflow(token: str) -> None:
    """Dispatch the auto-mossy workflow on the main branch."""

    url = (
        f"{GITHUB_API_BASE}/repos/{REPO}/actions/workflows/auto-mossy.yml/dispatches"
    )
    payload = {"ref": "main"}
    response = requests.post(
        url,
        headers=authorization_headers(token),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def handle_http_error(error: requests.exceptions.HTTPError) -> None:
    """Print a clear HTTP error message."""

    response = error.response
    if response is not None:
        try:
            message = response.json()
        except ValueError:
            message = response.text
        print(f"HTTP error {response.status_code}: {message}")
    else:
        print(f"HTTP error: {error}")


def main() -> None:
    pat = getpass("Paste your NEW GitHub PAT (ghp_…): ").strip()
    if not pat:
        print("No PAT provided.")
        sys.exit(1)

    otp_code = input("Paste the 8-digit verification code GitHub emailed you: ").strip()
    if not otp_code:
        print("No verification code provided.")
        sys.exit(1)

    masked_token = mask_pat(pat)

    try:
        finish_pat_authorization(pat, otp_code)
        public_key, key_id = get_repo_public_key(pat)
        encrypted_pat = encrypt_secret(pat, public_key)
        put_repo_secret(pat, encrypted_pat, key_id)
        dispatch_workflow(pat)
    except requests.exceptions.HTTPError as http_error:
        handle_http_error(http_error)
        print(f"Operation failed while using token {masked_token}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - provide clear CLI feedback
        print(f"Unexpected error: {exc}")
        print(f"Operation failed while using token {masked_token}")
        sys.exit(1)

    print("✅ PAT secret updated")
    print("✅ Workflow dispatched (check GitHub Actions tab)")


if __name__ == "__main__":
    main()
