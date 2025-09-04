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

import logging

def setup_logger():
  logging.basicConfig(level=logging.INFO)
  return logging.getLogger("moose_discussion_bot")
