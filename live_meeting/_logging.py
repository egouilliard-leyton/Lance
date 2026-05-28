"""Minimal stdlib logger for live_meeting.

NOTE: We intentionally do NOT reuse ``common.utils.logging.get_logger`` because it imports
``common.utils.distributed`` which imports ``torch`` at module load (common/utils/distributed.py:17).
The live_meeting package must stay CPU-only and import-light, so we use a self-contained
stdlib logger that mirrors the repo's format (minus the distributed/rank fields).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "[%(asctime)s][%(name)s][%(levelname).5s] %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name or "live_meeting")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    return logger
