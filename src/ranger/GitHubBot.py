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
import logging
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
from typing import List, Dict, Any, Tuple, Optional
from . import logger
from . import logger as logutil


class GitHubBot:
    def __init__(self, db_dir: Path, model_path: str, top_n: int, threshold: float, model_name: str, load_local: bool, dry_run: bool,logger: Optional[logging.Logger] = None, debug: bool = False) -> None:
        self.username = 'MOOSEbot'
        self.repo_owner = 'MengnanLi91'
        self.repo = os.getenv("GITHUB_REPO")
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
        self.db_dir = str(db_dir)

        # Logger setup
        self.log = logger if (logger is not None) else logutil.get_logger("ranger.githubbot")
        try:
            self.log.debug(
                "Initialized GitHubBot: repo=%s/%s top_n=%s threshold=%s dry_run=%s load_local=%s db_dir=%s",
                self.repo_owner, self.repo, self.top_n, self.threshold, self.dry_run, self.load_local, self.db_dir,
            )
        except Exception:
            pass


        # Load the embedding model
        if self.load_local:
            model_path_full = os.path.join(self.model_path, self.model_name)
            self.log.debug(f"Loading local model from {model_path_full}")
            self.embed_model = HuggingFaceEmbedding(model_name=model_path_full)
        else:
            self.log.debug("Loading model from HuggingFace")
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

    def generate_solution(self, title: str, top_n: int, index: SimpleVectorStore, threshold: float) -> Optional[str]:
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

        if not filtered_nodes:
            return None

        result: List[str] = []
        result.append(f"Here are some previous posts that may relate to your question: \n\n")

        for idx, node in enumerate(filtered_nodes):
            result.append(
                f"{idx + 1}. [{node.metadata['url']}]({node.metadata['url']})"
            )

        return "\n".join(result)

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
        if not self._ensure_owner_repo():
            return

        variables = {
            'owner': self.repo_owner,
            'repo': self.repo,
            'first': self.discussion_arr
        }

        headers = {"Authorization": f"bearer {self.github_token}"}

        if (not self.repo_owner or not self.repo) and os.getenv("GITHUB_REPO"):
            try:
                owner, repo = os.getenv("GITHUB_REPO").split("/", 1)
                self.repo_owner = self.repo_owner or owner
                self.repo = self.repo or repo
            except ValueError:
                pass

        if not self.github_token:
            self.log.error("GITHUB_TOKEN is missing.")
            return
        if not self.repo_owner or not self.repo:
            self.log.error("Repository not set (owner=%r repo=%r).", self.repo_owner, self.repo)
            return

        self.log.debug("POST %s variables=%s", self.end_point, variables)

        response = requests.post(self.end_point, json={'query': query, 'variables': variables}, headers=headers, verify=certifi.where())

        if response.status_code == 200:
            payload = self._parse_graphql(response)
            if not payload:
                return

            repo_obj = (payload.get('data') or {}).get('repository') or {}
            discussions = self._extract_discussion_nodes(repo_obj.get('discussions'))

            if not discussions:
                self.log.info("No discussions returned for %s/%s.", self.repo_owner, self.repo)
                return

            for discussion in discussions:
                title = discussion['title']
                author = discussion['author']['login']
                comments = discussion['comments']['nodes']

                if comments and comments[0]['author']['login'] != self.username:
                    continue

                concise_solution = self.generate_solution(title, self.top_n, self.index, self.threshold)
                if not concise_solution:
                    self.log.info(f"[skip] No similar results above threshold ({self.threshold}) for: {title!r}")
                    continue
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
                        self.log.info(f"Successfully replied to discussion: {title} (Comment ID: {comment_id})")
                    else:
                        self.log.error(f"Failed to add comment to discussion: {title}")
                else:
                    self.log.info(f"DRY RUN: Would have replied to discussion: '{title}' with the following body:\n{response_body}")
        else:
            self.log.error(f"Request failed with status code: {response.status_code}")
            self.log.debug(response.text)


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

        return sorted(best.values(), key=lambda n: getattr(n, "score", 0) or 0, reverse=True)

    # -------------------------------
    # Helpers: env → owner/repo, safe GraphQL parse, nodes extractor
    # -------------------------------
    def _ensure_owner_repo(self) -> bool:
        """
        Ensure self.repo_owner and self.repo are populated correctly.
        Accepts either:
        - GITHUB_REPOSITORY = "owner/repo"
        - GITHUB_OWNER + GITHUB_REPO
        - GITHUB_REPO = "repo" (owner from GITHUB_OWNER/GITHUB_REPOSITORY_OWNER)
        Returns True if both are set; logs an error and returns False otherwise.
        """
        # Derive from env if missing
        if not getattr(self, "repo_owner", None) or not getattr(self, "repo", None):
            repo_env = os.getenv("GITHUB_REPOSITORY") or os.getenv("GITHUB_REPO") or ""
            owner_env = os.getenv("GITHUB_OWNER") or os.getenv("GITHUB_REPOSITORY_OWNER")
            if "/" in (repo_env or ""):
                try:
                    owner, repo = repo_env.split("/", 1)
                    self.repo_owner = self.repo_owner or owner
                    self.repo = self.repo or repo
                except ValueError:
                    pass
            else:
                if repo_env and not getattr(self, "repo", None):
                    self.repo = repo_env
                if owner_env and not getattr(self, "repo_owner", None):
                    self.repo_owner = owner_env

        if not self.github_token:
            try:
                self.log.error("GITHUB_TOKEN is missing.");
            except Exception: pass
            return False

        if not self.repo_owner or not self.repo:
            try:
                self.log.error("Repository not set (owner=%r repo=%r). Set GITHUB_REPOSITORY or GITHUB_OWNER/GITHUB_REPO.", self.repo_owner, self.repo);
            except Exception:
                pass
            return False

        # Sanity: if repo accidentally includes a slash, split it
        if "/" in str(self.repo):
            try:
                self.log.warning("Repo name contains '/': %r. Splitting to owner/repo.", self.repo);
            except Exception:
                pass
            parts = str(self.repo).split("/", 1)
            self.repo_owner, self.repo = parts[0], parts[1]

        return True


    def _parse_graphql(self, response) -> Optional[Dict[str, Any]]:
        """Return parsed JSON payload or None; logs errors and never raises."""
        if response.status_code != 200:
            try:
                self.log.error("GraphQL request failed (%s): %s", response.status_code, response.text[:500]);
            except Exception: pass
            return None
        try:
            payload = response.json()
        except Exception:
            try:
                self.log.error("Response not JSON: %r", getattr(response, "text", "")[:500]);
            except Exception: pass
            return None

        if not isinstance(payload, dict):
            try:
                self.log.error("Unexpected payload type: %s", type(payload).__name__);
            except Exception: pass
            return None
        if payload.get("errors"):
            try:
                self.log.error("GraphQL errors: %s", payload["errors"]);
            except Exception: pass
            return None
        if not payload.get("data"):
            try:
                self.log.error("No 'data' in GraphQL response.");
            except Exception: pass
            return None

        return payload


    def _extract_discussion_nodes(self, discussions_obj: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Support both 'nodes' and 'edges{node}' shapes."""
        if not discussions_obj:
            return []
        nodes = discussions_obj.get("nodes")
        if nodes is not None:
            return nodes or []
        edges = discussions_obj.get("edges") or []
        out: List[Dict[str, Any]] = []
        for e in edges:
            if isinstance(e, dict) and isinstance(e.get("node"), dict):
                out.append(e["node"])
        return out
