# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
import sys, os
from pathlib import Path
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import RANGER

class TestGitHubBotCommand(unittest.TestCase):
    def test_run_bot_calls_query_response(self):
        cfg = RANGER.GitHubBotConfig(
            db_dir="./db",
            model_path="./models",
            model_name="all-MiniLM-L12-v2",
            top_n=3,
            threshold=0.1,
            load_local=True,
            dry_run=True,
        )
        with patch.object(RANGER, "GitHubBot") as MockBot:
            inst = MockBot.return_value
            inst.query_response = MagicMock()
            rc = RANGER.run_bot(cfg)
            self.assertEqual(rc, 0)
            inst.query_response.assert_called_once()

    def test_validation_offline_prefers_generate_solution(self):
        with tempfile.TemporaryDirectory() as td:
            val_out = Path(td) / "raw"
            val_db = Path(td) / "db"
            pin = Path(td) / "pinned.txt"
            pin.write_text("owner/repo#1\n")

            cfg_api = RANGER.GitHubAPIConfig(out_dir=str(val_out), dry_run=True)
            cfg_idx = RANGER.IndexGeneratorConfig(rawdata=str(val_out), database=str(val_db), dry_run=True)
            cfg_bot = RANGER.GitHubBotConfig(top_n=7, threshold=0.25)

            with patch.object(RANGER, "GitHubAPI") as MockAPI,                      patch.object(RANGER, "IndexGenerator") as MockIdx,                      patch.object(RANGER, "GitHubBot") as MockBot:

                api = MockAPI.return_value
                api.fetch_discussions_by_numbers = MagicMock()
                idx = MockIdx.return_value
                idx.generate_index = MagicMock()

                bot = MockBot.return_value
                bot.load_database = MagicMock(return_value=object())
                bot.generate_solution = MagicMock(return_value="answer")

                rc = RANGER.run_validation(
                    cfg_api, cfg_idx, cfg_bot,
                    prompt="What happened?",
                    pin_file=pin, golden_path=None, write_golden=None,
                    fail_on_mismatch=False,
                )
                self.assertEqual(rc, 0)
                bot.generate_solution.assert_called_once()

if __name__ == "__main__":
    unittest.main()
