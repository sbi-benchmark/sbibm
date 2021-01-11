import subprocess
from pathlib import Path
from typing import Optional


def commit_sha() -> Optional[str]:
    """Get current commit SHA

    Returns:
        Commit SHA if available
    """
    try:
        commit_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parents[1]
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        commit_sha = None

    return commit_sha
