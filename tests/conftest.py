from __future__ import annotations

import sys
import types
from pathlib import Path

# Ensure repository root is importable (e.g., `import src...`) regardless of
# how pytest is invoked in local/CI environments.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test environments may not have external dependencies pre-installed.
# Provide a minimal waitress shim so imports of `from waitress import serve`
# work during unit tests; production uses the real package from requirements.
if "waitress" not in sys.modules:
    waitress_stub = types.ModuleType("waitress")

    def _serve(*args, **kwargs):
        return None

    waitress_stub.serve = _serve
    sys.modules["waitress"] = waitress_stub
