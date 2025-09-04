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

from pathlib import Path
import requests
import os
import certifi
import argparse
from dotenv import load_dotenv
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    load_index_from_storage,
    QueryBundle,
    StorageContext,
    Settings,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List, Dict, Any, Tuple


class GitHubBot:
    def __init__(self, db_dir: Path, model_path: str, top_n: int, threshold: float, model_name: str, load_local: bool, dry_run: bool) -> None:
        self.username = 'MOOSEbot'
        self.repo_owner = 'MengnanLi91'
        self.repo = 'moose'
        self.end_point = "https://api.github.com/graphql"
        self.discussion_arr = 1  # Number of discussions to fetch
        load_dotenv()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.top_n = top_n
        self.threshold = threshold
        self.dry_run = dry_run
        self.load_local = load_local
        self.model_path = model_path
        self.model_name = model_name

        # Load the embedding model
        if self.load_local:
            model_path_full = os.path.join(self.model_path, self.model_name)
            print(f"Loading local model from {model_path_full}")
            self.embed_model = HuggingFaceEmbedding(model_name=model_path_full)
        else:
            print("Loading model from HuggingFace")
            self.embed_model = HuggingFaceEmbedding(model_name = f'sentence-transformers/{model_name}')

        self.index = None

    def load_database(self, db_dir: Path) -> SimpleVectorStore:
        Settings.embed_model = self.embed_model

        vector_store = SimpleVectorStore.from_persist_dir(db_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=db_dir
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index

    def generate_solution(self, title: str, top_n: int, index: SimpleVectorStore, threshold: float) -> str:
        if index is None:
            retrieved_nodes = VectorIndexRetriever.retrieve(None, QueryBundle(title))
        else:
            retriever = VectorIndexRetriever(
                index=index,
                similarity_metric="cosine",
                similarity_top_k=top_n,
                embed_model=self.embed_model,
            )
            retrieved_nodes = retriever.retrieve(QueryBundle(title))

        processor = SimilarityPostprocessor(similarity_cutoff=threshold)
        filtered_nodes = processor.postprocess_nodes(retrieved_nodes)
        filtered_nodes = self._deduplicate_nodes(filtered_nodes)

        result: List[str] = []
        result.append(f"Here are some previous posts that may relate to your question: \n\n")

        for idx, node in enumerate(filtered_nodes):
            result.append(
                f"{idx + 1}. [{node.metadata['url']}]({node.metadata['url']})"
            )

        return "\n".join(result)


    def _deduplicate_nodes(self, nodes: List[Any]) -> List[Any]:
        """
        Deduplicate by (metadata['url'], metadata['title']).
        Keeps the item with the highest .score for each pair.
        """
        def canon(s: str) -> str:
            # normalize whitespace + case; trim trailing slash/hash in URLs
            return " ".join((s or "").split()).strip().lower()

        def canon_url(u: str) -> str:
            u = (u or "").strip()
            # drop trailing "/" and fragment-only hashes to collapse trivial variants
            while u.endswith("/"):
                u = u[:-1]
            if "#" in u:
                u = u.split("#", 1)[0]
            return canon(u)

        best: Dict[Tuple[str, str], Any] = {}
        for nws in nodes or []:
            node = getattr(nws, "node", nws)
            meta = getattr(node, "metadata", {}) or {}
            key = (canon_url(meta.get("url", "")), canon(meta.get("title", "")))

            score = getattr(nws, "score", 0) or 0
            prev = best.get(key)
            prev_score = (getattr(prev, "score", 0) or 0) if prev is not None else None
            if prev is None or score > prev_score:
                best[key] = nws

        # Optional: sort by score, highest first
        return sorted(best.values(), key=lambda n: getattr(n, "score", 0) or 0, reverse=True)

    def query_response(self) -> None:

        if self.index is None:
            self.index = self.load_database(self.db_dir)

        query = '''
        query($owner: String!, $repo: String!, $first: Int!) {
        repository(owner: $owner, name: $repo) {
            discussions(first: $first) {
            totalCount
            pageInfo {
                hasNextPage
                endCursor
            }
            nodes {
                id
                title
                body
                author {
                login
                }
                comments(first: 1) {
                nodes {
                    author {
                    login
                    }
                    body
                }
                }
            }
            }
        }
        }
        '''

        mutation = '''
        mutation($discussionId: ID!, $body: String!) {
        addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
            comment {
            id
            }
        }
        }
        '''

        variables = {
            'owner': self.repo_owner,
            'repo': self.repo,
            'first': self.discussion_arr
        }

        headers = {"Authorization": f"bearer {self.github_token}"}

        response = requests.post(self.end_point, json={'query': query, 'variables': variables}, headers=headers, verify=certifi.where())

        if response.status_code == 200:
            data = response.json()
            discussions = data['data']['repository']['discussions']['nodes']

            for discussion in discussions:
                title = discussion['title']
                author = discussion['author']['login']
                comments = discussion['comments']['nodes']

                if comments and comments[0]['author']['login'] != self.username:
                    continue

                concise_solution = self.generate_solution(title, self.top_n, self.index, self.threshold)
                response_body = (
                    f"Hey, @{author},\n\n"
                    f"{concise_solution}\n\n"
                    "Note: This is an automated response. Please review and verify the solution.\n"
                    f"@{self.username} [BOT]"
                )

                discussion_id = discussion['id']

                if not self.dry_run:
                    response = requests.post(self.end_point, json={'query': mutation, 'variables': {'discussionId': discussion_id, 'body': response_body}}, headers=headers)

                    if response.status_code == 200:
                        response_data = response.json()
                        comment_id = response_data['data']['addDiscussionComment']['comment']['id']
                        print(f"Successfully replied to discussion: {title} (Comment ID: {comment_id})")
                    else:
                        print(f"Failed to add comment to discussion: {title}")
                else:
                    print(f"Dry run mode: Would have replied to discussion: '{title}' with the following body:\n{response_body}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)


