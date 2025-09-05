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

import argparse
from pathlib import Path
import requests
import json
from dotenv import load_dotenv
import os
import certifi
from typing import Optional, Dict, Any

class GitHubAPI:
    def __init__(self, end_point: str, num_discussion: int, num_comment: int, num_reply: int, min_credit: int, out_dir: str, dry_run: bool) -> None:
        """
        Initialize the GitHubAPI class with endpoint, query parameters, output directory, and dry run mode.

        Parameters:
        - end_point (str): The GitHub GraphQL API endpoint.
        - num_discussion (int): Number of discussions to retrieve.
        - num_comment (int): Number of comments per discussion to retrieve.
        - num_reply (int): Number of replies per comment to retrieve.
        - min_credit (int): Minimum credit required to continue making API requests.
        - out_dir (str): Directory to save the response files.
        - dry_run (bool): Whether to run the script in dry run mode.
        """
        self.end_point: str = end_point
        self.num_discussion: int = num_discussion
        self.num_comment: int = num_comment
        self.num_reply: int = num_reply
        self.min_credit: int = min_credit
        self.out_dir: Path = Path(out_dir)
        self.STATUS_SUCCESS: int = 200
        self.has_remaining_credit: bool = True
        self.has_next_page: bool = True
        self.end_cursor: str = "null"
        self.dry_run: bool = dry_run
        load_dotenv()
        self.query_template: str = Path("query.gql.in").read_text()
        self.GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
        self.headers: dict = {"Authorization": f"bearer {self.GITHUB_TOKEN}"}
        self.out_dir.mkdir(exist_ok=True)

    def log(self, begin_cursor: str, end_cursor: str, remaining: int, has_next_page: bool) -> None:
        """
        Log the details of the current API request.

        Parameters:
        - begin_cursor (str): The cursor position before the current request.
        - end_cursor (str): The cursor position after the current request.
        - remaining (int): Remaining API request credits.
        - has_next_page (bool): Whether there are more pages to fetch.
        """
        print("            From: {}".format(begin_cursor))
        print("              To: {}".format(end_cursor))
        print("Remaining credit: {}".format(remaining))
        print("        Has more: {}".format(has_next_page))
        print("-" * 79)

    def fetch_data(self) -> None:
        """
        Fetch data from the GitHub GraphQL API, paginate through results, and save responses to files.
        """
        while self.has_next_page and self.has_remaining_credit:
            query: str = self.query_template.format(
                self.end_cursor,
                self.num_discussion,
                self.num_comment,
                self.num_reply,
            )
            if self.dry_run:
                print("Dry run: would execute query with cursor {}".format(self.end_cursor))
                break

            response: requests.Response = requests.post(self.end_point, headers=self.headers, json={"query": query}, verify=certifi.where())
            if response.status_code == self.STATUS_SUCCESS:
                result: dict = response.json()
                if "data" not in result:
                    print("Error: 'data' key not found in the response.")
                    print(result)
                    exit()

                begin_cursor: str = self.end_cursor
                page_info: dict = result["data"]["repository"]["discussions"]["pageInfo"]
                self.has_next_page = page_info["hasNextPage"]
                self.end_cursor = '"' + page_info["endCursor"] + '"'
                remaining: int = result["data"]["rateLimit"]["remaining"]
                self.has_remaining_credit = remaining >= self.min_credit
                self.log(begin_cursor, self.end_cursor, remaining, self.has_next_page)
                out_file: Path = self.out_dir / "{}_{}.json".format(begin_cursor, self.end_cursor)

                repository = result["data"]["repository"]

                self._attach_discussion_category_metadata(repository)

                with out_file.open("w") as file:
                    json.dump(repository, file, indent=2)
            else:
                print("Error: {}".format(response.status_code))
                print(response.text)
                exit()

        if not self.has_next_page:
            print("All discussions have been fetched.")

    def _attach_discussion_category_metadata(self, repository_payload: Dict[str, Any]) -> None:
        """
        Copy node.category.name into node.metadata.discussion_category_name
        so downstream code can read it without depending on GraphQL shape.
        """
        try:
            edges = (repository_payload.get("discussions") or {}).get("edges") or []
            for edge in edges:
                node = (edge or {}).get("node") or {}
                cat = node.get("category") or {}
                name = cat.get("name")
                if not name:
                    continue
                meta = dict(node.get("metadata") or {})
                meta["discussion_category_name"] = name
                node["metadata"] = meta
        except Exception as e:
            # non-fatal; we still want to write raw payload
            print(f"Warning: unable to attach DiscussionCategory metadata: {e}")

    def _post_graphql(self, query: str, variables: dict) -> dict:
        """Send a GraphQL POST request to GitHub and return the parsed JSON.

        Respects:
            - self.end_point (str): GraphQL endpoint
            - self.headers (Dict[str, str]): HTTP headers (e.g., Authorization)
            - self.STATUS_SUCCESS (int): success HTTP code (defaults to 200)
            - self.dry_run (bool): if True, do not make the request
        """
        if getattr(self, "dry_run", False):
            print(f"[DRY RUN] Would POST GraphQL with variables={variables}")
            return {}
        response = requests.post(
            self.end_point,
            headers=getattr(self, "headers", {}),
            json={"query": query, "variables": variables},
            verify=certifi.where(),
        )
        status_success = getattr(self, "STATUS_SUCCESS", 200)
        if response.status_code != status_success:
            raise RuntimeError(f"GitHub GraphQL error: HTTP {response.status_code} → {response.text}")
        data = response.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(f"GitHub GraphQL returned errors: {data['errors']}")
        return data

    def fetch_discussion_node_by_number(self, owner: str, repo: str, number: int) -> dict:
        """Fetch a single discussion by number and return the discussion node.

        The returned dict includes the minimal fields typically needed downstream:
        - number, title, url, bodyText
        - comments.edges[].node.bodyText
        - comments.edges[].node.replies.edges[].node.bodyText
        """
        query = """
        query($owner: String!, $repo: String!, $number: Int!,
            $numComment: Int!, $numReply: Int!) {
        repository(owner: $owner, name: $repo) {
            discussion(number: $number) {
            number
            title
            url
            bodyText
            comments(first: $numComment) {
                edges {
                node {
                    bodyText
                    replies(first: $numReply) {
                    edges {
                        node { bodyText }
                    }
                    }
                }
                }
            }
            }
        }
        rateLimit { remaining resetAt }
        }
        """
        variables = {
            "owner": owner,
            "repo": repo,
            "number": int(number),
            "numComment": int(getattr(self, "num_comment", 50)),
            "numReply": int(getattr(self, "num_reply", 50)),
        }
        data = self._post_graphql(query, variables)
        if getattr(self, "dry_run", False):
            return {
                "number": number,
                "title": f"[DRY RUN] Discussion #{number}",
                "url": f"https://github.com/{owner}/{repo}/discussions/{number}",
                "bodyText": "",
                "comments": {"edges": []},
            }
        repo_obj = data.get("data", {}).get("repository")
        if not repo_obj or not repo_obj.get("discussion"):
            raise RuntimeError(f"Discussion #{number} not found in {owner}/{repo}")
        return repo_obj["discussion"]

    def fetch_discussions_by_numbers(self, owner: str, repo: str, numbers: list[int], out_dir: str | None = None):
        """Deterministically fetch an exact set of discussions and write one JSON page.

        The written file shape matches common readers expecting:
            {
            "discussions": {
                "edges": [ {"node": <discussion>}, ... ]
            }
            }
        """
        dest_dir = Path(out_dir) if out_dir is not None else Path(getattr(self, "out_dir", "./out"))
        dest_dir.mkdir(parents=True, exist_ok=True)

        edges = []
        for n in numbers:
            node = self.fetch_discussion_node_by_number(owner, repo, int(n))
            edges.append({"node": node})

        page = {"discussions": {"edges": edges}}
        numbers_part = "-".join(str(int(n)) for n in numbers)
        out_path = dest_dir / f"pinned_{owner}_{repo}_{numbers_part}.json"

        if getattr(self, "dry_run", False):
            print(f"[DRY RUN] Would write pinned page to: {out_path}")
            return out_path

        out_path.write_text(json.dumps(page, ensure_ascii=False, indent=2))
        print(f"Wrote pinned discussions page → {out_path}")
        return out_path
