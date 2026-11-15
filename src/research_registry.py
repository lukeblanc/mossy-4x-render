"""Import the reviewed experiment catalogue into private research memory only.

This module has no broker, strategy, promotion or order API. Importing a proposal
never supplies a result or makes its historical source claims verified.
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CATALOGUE = Path(__file__).resolve().parents[1] / "research" / "market_experiments_2026-11-14.json"


def publish_registry(data_dir, source=CATALOGUE):
    try:
        if os.getenv("MODE", "").lower() != "demo" or os.getenv("OANDA_ENV", "").lower() != "practice":
            raise ValueError("practice-demo-mode-required")
        raw = Path(source).read_bytes()
        catalogue = json.loads(raw)
        if catalogue.get("schema") != "mossy.research-catalogue.v1":
            raise ValueError("unsupported-catalogue-schema")
        experiments = catalogue["experiments"]
        ids = [e["id"] for e in experiments]
        if not experiments or len(set(ids)) != len(ids):
            raise ValueError("missing-or-duplicate-experiment")
        if any(e.get("status") != "PROPOSED" or e.get("results") is not None
               or not e.get("required_data") or not e.get("validation") for e in experiments):
            raise ValueError("proposal-cannot-claim-validation")
        report = {"schema": "mossy.research-registry.v1", "status": "IMPORTED_PROPOSALS",
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "catalogue_sha256": hashlib.sha256(raw).hexdigest(),
                  "execution_enabled": False, "promotion_enabled": False,
                  "catalogue": catalogue}
        folder = Path(data_dir) / "memory" / "research"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(folder, 0o700)
        fd, temporary = tempfile.mkstemp(prefix=".registry-", dir=folder)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(report, output, sort_keys=True, indent=2, allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, folder / "registry.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print("[RESEARCH_REGISTRY] " + json.dumps({
            "status": report["status"], "catalogue_sha256": report["catalogue_sha256"],
            "experiment_count": len(experiments), "experiment_ids": ids,
            "validated_experiments": 0, "execution_enabled": False,
            "generated_at_utc": report["generated_at_utc"]}), flush=True)
        return report
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("[RESEARCH_REGISTRY] " + json.dumps({"status": "UNAVAILABLE", "reason": type(exc).__name__}), flush=True)
        return None
