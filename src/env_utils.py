import os
from pathlib import Path


def load_repo_env(env_file: str = ".env") -> None:
    """Load KEY=VALUE pairs from the repo root .env into os.environ if unset."""
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    env_path = repo_root / env_file

    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
