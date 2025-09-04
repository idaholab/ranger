# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

import sys, os
# Ensure the project root (where RANGER.py lives) is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import RANGER  # the CLI/driver

class TestGitHubAPICommand(unittest.TestCase):
    def test_run_github_api_calls_fetch(self):
        cfg = RANGER.GitHubAPIConfig(
            end_point="https://api.github.com/graphql",
            num_discussion=1,
            num_comment=1,
            num_reply=1,
            min_credit=100,
            out_dir="./out",
            dry_run=True,
        )
        with patch.object(RANGER, "GitHubAPI") as MockAPI:
            inst = MockAPI.return_value
            inst.fetch_data = MagicMock()
            rc = RANGER.run_github_api(cfg)
            self.assertEqual(rc, 0)
            inst.fetch_data.assert_called_once()

    def test_validation_uses_pin_and_golden(self):
        # Integration-style test of run_validation wiring with mocks
        with tempfile.TemporaryDirectory() as td:
            val_out = Path(td) / "raw"
            val_db = Path(td) / "db"
            golden = Path(td) / "golden.txt"
            pin = Path(td) / "pinned.txt"
            pin.write_text("owner/repo#123\n")

            # Prepare configs
            cfg_api = RANGER.GitHubAPIConfig(out_dir=str(val_out), dry_run=True)
            cfg_idx = RANGER.IndexGeneratorConfig(rawdata=str(val_out), database=str(val_db), dry_run=True)
            cfg_bot = RANGER.GitHubBotConfig()

            # Mocks
            with patch.object(RANGER, "GitHubAPI") as MockAPI, patch.object(RANGER, "IndexGenerator") as MockIdx,                      patch.object(RANGER, "GitHubBot") as MockBot:

                api = MockAPI.return_value
                api.fetch_discussions_by_numbers = MagicMock()
                api.fetch_data = MagicMock()

                idx = MockIdx.return_value
                idx.generate_index = MagicMock()

                bot = MockBot.return_value
                # Bot offline path
                bot.load_database = MagicMock(return_value=object())
                bot.generate_solution = MagicMock(return_value="hello world")

                rc = RANGER.run_validation(
                    cfg_api, cfg_idx, cfg_bot, prompt="hi",
                    pin_file=pin, golden_path=None, write_golden=golden,
                    fail_on_mismatch=False
                )
                self.assertEqual(rc, 0)
                # PIN path should be used, not generic fetch
                api.fetch_discussions_by_numbers.assert_called_once()
                idx.generate_index.assert_called_once()
                bot.generate_solution.assert_called_once()
                self.assertEqual(golden.read_text(), "hello world")

if __name__ == "__main__":
    unittest.main()
