"""Tools for encrypting and pushing patch diffs to GitHub secrets."""
from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

import requests
from nacl import encoding, public


GITHUB_API_BASE = "https://api.github.com"
SECRET_NAME = "PATCH_BLOB"
REQUEST_TIMEOUT = 30


class PatchError(Exception):
    """Raised when updating the patch secret fails."""


def encrypt_secret(value: str, public_key: str) -> str:
    """Encrypt ``value`` using GitHub's repository ``public_key``.

    Args:
        value: The plaintext secret to encrypt.
        public_key: Base64 encoded public key string from GitHub.

    Returns:
        A base64 encoded ciphertext suitable for GitHub secrets API.
    """

    key = public.PublicKey(public_key, encoding.Base64Encoder())
    sealed_box = public.SealedBox(key)
    encrypted = sealed_box.encrypt(value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def get_repo_public_key(repo: str, token: str) -> Tuple[str, str]:
    """Fetch the repository's public key and key ID."""

    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/secrets/public-key"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload["key"], payload["key_id"]


def update_patch_secret(repo: str, token: str, encrypted_value: str, key_id: str) -> None:
    """Create or update the PATCH_BLOB secret with ``encrypted_value``."""

    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/secrets/{SECRET_NAME}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"encrypted_value": encrypted_value, "key_id": key_id}
    response = requests.put(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()


def trigger_render_deploy(url: str) -> None:
    """Trigger a Render deploy hook if ``url`` is provided."""

    response = requests.post(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()


def push_patch(diff_content: Optional[str] = None) -> None:
    """Encrypt and push the diff content as a GitHub secret."""

    repo = os.getenv("GITHUB_REPO")
    if not repo:
        raise PatchError("GITHUB_REPO is not set")

    token = os.getenv("GH_PAT")
    if not token:
        raise PatchError("GH_PAT is not set")

    if diff_content is None:
        diff_content = os.getenv("DIFF_CONTENT")
    if not diff_content:
        raise PatchError("DIFF_CONTENT is not set or empty")

    public_key, key_id = get_repo_public_key(repo, token)
    encrypted_value = encrypt_secret(diff_content, public_key)
    update_patch_secret(repo, token, encrypted_value, key_id)

    deploy_hook_url = os.getenv("RENDER_DEPLOY_HOOK_URL")
    if deploy_hook_url:
        trigger_render_deploy(deploy_hook_url)


def load_env_if_available() -> None:
    """Load environment variables from a .env file when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv  # type: ignore import-not-found
    except ModuleNotFoundError:
        return
    load_dotenv()


def main() -> None:
    try:
        load_env_if_available()
        push_patch()
    except Exception as exc:  # noqa: BLE001 - allow broad exception for CLI feedback
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    print("PATCH OK")


if __name__ == "__main__":
    main()
