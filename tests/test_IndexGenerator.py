# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import RANGER

class TestIndexGeneratorCommand(unittest.TestCase):
    def test_run_index_calls_generate(self):
        cfg = RANGER.IndexGeneratorConfig(
            load_local=True,
            model_path="./models",
            model_name="all-MiniLM-L12-v2",
            show_progress=False,
            rawdata="./raw",
            database="./db",
            dry_run=True,
        )
        with patch.object(RANGER, "IndexGenerator") as MockIdx:
            inst = MockIdx.return_value
            inst.generate_index = MagicMock()
            rc = RANGER.run_index(cfg)
            self.assertEqual(rc, 0)
            inst.generate_index.assert_called_once()

if __name__ == "__main__":
    unittest.main()