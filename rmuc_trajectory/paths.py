"""Portable output configuration shared by replay entry points."""

import json
import os
from pathlib import Path


def get_output_root(root: Path | None = None, environ=None) -> Path:
    """Use environment, ignored project-local config, then project outputs.

    CLI --output-dir arguments override this root in the individual entry points.
    An explicitly configured location is never replaced after an I/O failure.
    """
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    environ = os.environ if environ is None else environ
    configured = environ.get("BATTLESCOPE_OUTPUT_DIR", "").strip()
    if not configured:
        local = root / ".battlescope.local.json"
        if local.is_file():
            settings = json.loads(local.read_text(encoding="utf-8-sig"))
            configured = settings.get("output_dir", "")
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError(".battlescope.local.json requires a nonempty output_dir string")
    path = Path(configured).expanduser() if configured else root / "outputs"
    return path.resolve() if path.is_absolute() else (root / path).resolve()


DEFAULT_OUTPUT_ROOT = get_output_root()
