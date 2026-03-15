"""Auto-start Qdrant Docker container if not already running."""

import logging
import subprocess
from urllib.error import URLError
from urllib.request import urlopen

from config.settings import settings

logger = logging.getLogger(__name__)

_HEALTHZ_TIMEOUT_S = 3
_POST_START_RETRIES = 10


def _is_healthy() -> bool:
    """Ping Qdrant /healthz endpoint."""
    url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/healthz"
    try:
        with urlopen(url, timeout=_HEALTHZ_TIMEOUT_S) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def ensure_qdrant_running() -> None:
    """Ensure the Qdrant Docker container is running.

    If Qdrant is already reachable, this is a no-op.
    Otherwise, runs ``scripts/setup_qdrant.sh start`` and verifies health.
    """
    if _is_healthy():
        logger.info("Qdrant is already running.")
        return

    script = settings.PROJECT_ROOT / "scripts" / "setup_qdrant.sh"
    logger.info("Qdrant not reachable — starting via %s ...", script)

    result = subprocess.run(
        [str(script), "start"],
        cwd=str(settings.PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"setup_qdrant.sh start failed (rc={result.returncode}):\n{result.stderr}"
        )

    # The script already waits for health, but double-check from Python side
    import time

    for _ in range(_POST_START_RETRIES):
        if _is_healthy():
            logger.info("Qdrant is now running.")
            return
        time.sleep(1)

    raise RuntimeError(
        "Qdrant is still unreachable after setup_qdrant.sh start completed."
    )
