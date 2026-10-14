"""Configuration loading for swarm agents."""

import logging
from dataclasses import dataclass, field

from pathlib import Path

from dotenv import load_dotenv  # type: ignore  # noqa: F401 (re-exported for patching)

# Root of the hollow_mirror package (where .env, agent.md, tools.json, prompts/ live)
PACKAGE_ROOT = Path(__file__).parent
SOCIAL_DEDUCTION_PATH = PACKAGE_ROOT / "social_deduction"

logger = logging.getLogger(__name__)


def load_env(env_path: Path | None = None) -> None:
    """Load the hollow_mirror .env file into os.environ.

    Uses python-dotenv with interpolation enabled so that variables like
    ``${DB_CONNECTION_USER}`` in the file are expanded before being set.

    Existing environment variables are NOT overridden (override=False),
    so shell exports always win over .env values.

    Args:
        env_path: Explicit path to the .env file. Defaults to
                  ``hollow_mirror/.env`` (alongside this module).
    """
    path = env_path or (PACKAGE_ROOT / ".env")
    if not path.exists():
        logger.warning(".env not found at %s — relying on shell environment", path)
        return

    loaded = load_dotenv(dotenv_path=path, override=False, interpolate=True)
    if loaded:
        logger.info("Loaded .env from %s", path)
    else:
        logger.debug(".env at %s produced no new variables (already set?)", path)
