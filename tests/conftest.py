from __future__ import annotations

import sys
import types

# Test environments may not have external dependencies pre-installed.
# Provide a minimal waitress shim so imports of `from waitress import serve`
# work during unit tests; production uses the real package from requirements.
if "waitress" not in sys.modules:
    waitress_stub = types.ModuleType("waitress")

    def _serve(*args, **kwargs):
        return None

    waitress_stub.serve = _serve
    sys.modules["waitress"] = waitress_stub
