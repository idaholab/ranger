# -----------------------------------------------------------------------------
# This file is part of RANGER
#
# A Python‑based auto‑response bot to monitor and generate relevant responses
# for new discussions in the GitHub MOOSE repository.
#
# Licensed under the MIT License; see LICENSE for details:
#     https://spdx.org/licenses/MIT.html
#
# Copyright (c) 2025 Battelle Energy Alliance, LLC.
# All Rights Reserved.
# -----------------------------------------------------------------------------

import os
import logging

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def get_logger(name: str = "ranger.githubbot") -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "DEBUG" if _env_flag("DEBUG") else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)

    logging.getLogger("urllib3").setLevel(logging.WARNING if level > logging.DEBUG else logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING if level > logging.DEBUG else logging.DEBUG)
    return logger

