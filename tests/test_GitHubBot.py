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


import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from RANGER import GitHubBot
import os
import requests


class TestGitHubBot(unittest.TestCase):
    def setUp(self):

        self.db_dir = Path("tests/testdatabase")
        self.model_path = Path("models")
        self.top_n = 5
        self.threshold = 0.2
        self.model_name = "all-MiniLM-L12-v2"
        self.load_local = True
        self.dry_run = True

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model path '{self.model_path}' does not exist, please downloaded f{self.model_name} locally to run within INL network."
            )

        self.bot = GitHubBot(
            db_dir=self.db_dir,
            model_path=self.model_path,
            top_n=self.top_n,
            threshold=self.threshold,
            model_name=self.model_name,
            load_local=self.load_local,
            dry_run=self.dry_run,
        )

    @patch("moose_discussion_bot.GitHubBot.HuggingFaceEmbedding")
    @patch("moose_discussion_bot.GitHubBot.SimpleVectorStore.from_persist_dir")
    @patch("moose_discussion_bot.GitHubBot.StorageContext.from_defaults")
    @patch("moose_discussion_bot.GitHubBot.load_index_from_storage")
    def test_load_database(
        self,
        mock_load_index_from_storage,
        mock_storage_context,
        mock_from_persist_dir,
        mock_embedding,
    ):
        mock_index = MagicMock()
        mock_load_index_from_storage.return_value = mock_index

        index = self.bot.load_database(self.db_dir)

        mock_from_persist_dir.assert_called_once_with(self.db_dir)
        mock_storage_context.assert_called_once()
        mock_load_index_from_storage.assert_called_once_with(
            storage_context=mock_storage_context()
        )
        self.assertEqual(index, mock_index)

    @patch("moose_discussion_bot.GitHubBot.VectorIndexRetriever.retrieve")
    @patch("moose_discussion_bot.GitHubBot.SimilarityPostprocessor.postprocess_nodes")
    def test_generate_solution(self, mock_postprocess_nodes, mock_retrieve):
        title = "Sample Title"
        mock_node = MagicMock()
        mock_node.metadata = {"url": "http://example.com"}
        mock_node.score = 0.8
        mock_retrieve.return_value = [mock_node]
        mock_postprocess_nodes.return_value = [mock_node]

        solution = self.bot.generate_solution(
            title, self.top_n, self.bot.index, self.threshold
        )

        self.assertIn(
            "Here are some previous posts that may relate to your question:", solution
        )
        self.assertIn("1. [http://example.com](http://example.com)", solution)

    @patch("moose_discussion_bot.GitHubBot.requests.post")
    def test_query_response(self, mock_post):
        mock_query_response = MagicMock()
        mock_query_response.status_code = 200
        mock_query_response.json.return_value = {
            "data": {
                "repository": {
                    "discussions": {
                        "nodes": [
                            {
                                "id": "1",
                                "title": "Sample Discussion Title",
                                "body": "Sample body",
                                "author": {"login": "sample_user"},
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "MOOSEbot"},
                                            "body": "Sample comment",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            }
        }
        # Mock the mutation response
        mock_mutation_response = MagicMock()
        mock_mutation_response.status_code = 200
        mock_mutation_response.json.return_value = {
            "data": {"addDiscussionComment": {"comment": {"id": "comment_id"}}}
        }

        mock_post.side_effect = [mock_query_response, mock_mutation_response]

        with patch.object(
            self.bot, "generate_solution", return_value="Sample Solution"
        ):
            with patch("builtins.print") as mock_print:
                self.bot.query_response()
                mock_print.assert_called_with(
                    "Dry run mode: Would have replied to discussion: 'Sample Discussion Title' with the following body:\nHey, @sample_user,\n\nSample Solution\n\nNote: This is an automated response. Please review and verify the solution.\n@MOOSEbot [BOT]"
                )

    @patch("moose_discussion_bot.GitHubBot.requests.post")
    @patch("moose_discussion_bot.GitHubBot.SimpleVectorStore.from_persist_dir")
    def test_query_response_failure(self, mock_from_persist_dir, mock_post):
        # Mock the vector store loading
        mock_from_persist_dir.return_value = MagicMock()

        # Mock a failed response from GitHub API
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Capture the print output
        with patch("builtins.print") as mocked_print:
            self.bot.query_response()
            mocked_print.assert_any_call("Request failed with status code: 500")
            mocked_print.assert_any_call("Internal Server Error")

        # Check if the requests.post was called
        self.assertTrue(mock_post.called, "requests.post should be called")


if __name__ == "__main__":
    unittest.main()
