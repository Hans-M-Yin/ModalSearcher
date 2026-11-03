"""Load simple shell environment files for runtime utilities."""

import os
from pathlib import Path
import shlex


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load simple `export KEY=value` shell env files without extra deps."""

    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_no} is not KEY=value syntax: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no} has empty env key.")
        parsed = shlex.split(value, comments=False, posix=True)
        env_value = parsed[0] if parsed else ""
        if override or key not in os.environ:
            os.environ[key] = env_value
        loaded[key] = env_value
    return loaded
