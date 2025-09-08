# -----------------------------------------------------------------------------
# This file is main file of RANGER
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

#!/usr/bin/env python3

"""
Example minimal YAML (config.yaml):

---
# Default values applied to all sections unless overridden
default:
  dry_run: false

# GitHub API fetch config
github_api:
  end_point: "https://api.github.com/graphql"
  num_discussion: 10
  num_comment: 50
  num_reply: 50
  min_credit: 100
  out_dir: "./out"
  dry_run: false

# Indexing config
index:
  load_local: false
  model_path: "./models/"
  model_name: "all-MiniLM-L6-v2"
  show_progress: true
  rawdata: "./rawdata"
  database: "./database"
  dry_run: false

bot:
  db_dir: "./database"
  model_path: "./models/"
  model_name: "all-MiniLM-L6-v2"
  top_n: 5
  threshold: 0.3
  load_local: false
  dry_run: false
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Dict, Optional
import difflib
import os

try:
    import yaml
except Exception:
    yaml = None

import inspect

# -----------------------------------------------------------------------------
# Import project modules
# -----------------------------------------------------------------------------

# Locate the project's src/ next to RANGER.py
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now you can import modules that live directly under src/
from ranger import GitHubAPI, IndexGenerator, GitHubBot, utils


# -------------------------------
# Configuration dataclasses
# -------------------------------
@dataclass
class CommonConfig:
    dry_run: bool = False


@dataclass
class GitHubAPIConfig(CommonConfig):
    end_point: str = "https://api.github.com/graphql"
    num_discussion: int = 10
    num_comment: int = 50
    num_reply: int = 50
    min_credit: int = 100
    out_dir: str = "./out"


@dataclass
class IndexGeneratorConfig(CommonConfig):
    load_local: bool = False
    model_path: str = "./models/"
    model_name: str = "all-MiniLM-L12-v2"
    show_progress: bool = True
    rawdata: str = "./rawdata"
    database: str = "./database"


@dataclass
class GitHubBotConfig(CommonConfig):
    db_dir: str = "./database"
    model_path: str = "./models/"
    model_name: str = "all-MiniLM-L12-v2"
    top_n: int = 5
    threshold: float = 0.3
    load_local: bool = False


@dataclass
class AppConfig:
    default: CommonConfig = field(default_factory=CommonConfig)
    github_api: GitHubAPIConfig = field(default_factory=GitHubAPIConfig)
    index: IndexGeneratorConfig = field(default_factory=IndexGeneratorConfig)
    bot: GitHubBotConfig = field(default_factory=GitHubBotConfig)

    @staticmethod
    def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = AppConfig._deep_update(out[k], v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_yaml(cls, path: Optional[Path]) -> "AppConfig":
        if not path or not yaml:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        # Merge defaults per section with top-level defaults applied
        def apply(section_cls, section_name: str):
            section = getattr(cfg, section_name)
            incoming = (data.get(section_name) or {})
            merged = {**asdict(cfg.default), **asdict(section), **incoming}
            # Filter unknown keys
            field_names = {f.name for f in section_cls.__dataclass_fields__.values()}
            merged = {k: v for k, v in merged.items() if k in field_names}
            setattr(cfg, section_name, section_cls(**merged))
        apply(GitHubAPIConfig, "github_api")
        apply(IndexGeneratorConfig, "index")
        apply(GitHubBotConfig, "bot")

        return cfg

    def override_with_cli(self, args: argparse.Namespace, command: str) -> "AppConfig":
        # Copy and override according to the active subcommand
        cfg = AppConfig(
            default=self.default,
            github_api=self.github_api,
            index=self.index,
            bot=self.bot,
        )
        # Apply global dry_run if set (only stored under default)
        if getattr(args, "dry_run", False):
            cfg.default = CommonConfig(dry_run=True)
        target = getattr(cfg, {"github-api": "github_api", "index": "index", "bot": "bot"}.get(command, "default"))
        for k in asdict(target).keys():
            if getattr(args, k, None) is not None:
                setattr(target, k, getattr(args, k))
        return cfg

# -------------------------------
# CLI parser
# -------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="RANGER", description="Unified CLI for GitHub data, indexing, and bot operations.")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--print-config", action="store_true", help="Print the final merged config for the subcommand and exit.")
    subs = p.add_subparsers(dest="command", required=True)

    # github-api
    pa = subs.add_parser("github-api", help="Fetch data from GitHub GraphQL API.")
    pa.add_argument("--end-point", dest="end_point", type=str, help="GitHub GraphQL API endpoint.")
    pa.add_argument("--num-discussion", dest="num_discussion", type=int, help="Number of discussions to retrieve.")
    pa.add_argument("--num-comment", dest="num_comment", type=int, help="Number of comments per discussion.")
    pa.add_argument("--num-reply", dest="num_reply", type=int, help="Number of replies per comment.")
    pa.add_argument("--min-credit", dest="min_credit", type=int, help="Minimum credit to continue making API requests.")
    pa.add_argument("--out-dir", dest="out_dir", type=str, help="Directory to save response files.")
    pa.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode.")
    pa.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging.")

    # index
    pi = subs.add_parser("index", help="Generate a vector DB from raw GitHub JSON data.")
    pi.add_argument("--load-local", dest="load_local", action="store_true", help="Load a local model path.")
    pi.add_argument("--model-path", dest="model_path", type=str, help="Local model path.")
    pi.add_argument("--model-name", dest="model_name", type=str, help="Model name (HF ID or local dir).")
    pi.add_argument("--show-progress", dest="show_progress", action="store_true", help="Show progress bar.")
    pi.add_argument("--rawdata", dest="rawdata", type=str, help="Input data folder path.")
    pi.add_argument("--database", dest="database", type=str, help="Output index database path.")
    pi.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode.")
    pi.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging.")

    # bot
    pb = subs.add_parser("bot", help="Run the bot against a vector DB.")
    pb.add_argument("--db-dir", dest="db_dir", type=str, help="Index data folder path.")
    pb.add_argument("--model-path", dest="model_path", type=str, help="Local model path.")
    pb.add_argument("--model-name", dest="model_name", type=str, help="Model name (HF ID or local dir).")
    pb.add_argument("--top-n", dest="top_n", type=int, help="The number of suggestions, range 1-10.")
    pb.add_argument("--threshold", dest="threshold", type=float, help="Relevance of suggestion, less than 1.0.")
    pb.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode.")
    pb.add_argument("--load-local", dest="load_local", action="store_true", help="Load a local model path.")
    pb.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging.")

    # validation
    pv = subs.add_parser("validation", help="End-to-end validation: fetch 5 discussions, build a temp vector DB, run the bot.")
    pv.add_argument("--val-out-dir", dest="val_out_dir", type=str, default="./validation/raw", help="Directory to save raw validation data pulled from GitHub.")
    pv.add_argument("--val-db", dest="val_db", type=str, default="./validation_db", help="Directory for the validation vector database.")
    pv.add_argument("--prompt", dest="prompt", type=str, default="Shallow Water Equations", help="Optional prompt to pass to the bot for a one-off response.")
    pv.add_argument("--load-local", dest="load_local", action="store_true", help="Load local embedding model from disk for indexing/bot (overrides YAML).")
    pv.add_argument("--model-path", dest="model_path", type=str, default=None, help="Local model path override for indexing/bot.")
    pv.add_argument("--model-name", dest="model_name", type=str, default=None, help="Model name override for indexing/bot.")
    pv.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode for all steps.")
    pv.add_argument("--pin-file", dest="pin_file", type=Path, default="pinned.txt", help="Text file with owner/repo on first line and discussion numbers below.")
    pv.add_argument("--golden", dest="golden_path", type=Path, default="golden.txt", help="Path to an existing golden file to compare against.")
    pv.add_argument("--write-golden", dest="write_golden", type=Path, help="Write the produced output to this golden file (overwrites).")
    pv.add_argument("--fail-on-mismatch", dest="fail_on_mismatch", action="store_true", help="Exit non-zero if comparison fails.")
    pv.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging.")

    return p


# -------------------------------
# Entrypoints
# -------------------------------

def run_github_api(cfg: GitHubAPIConfig) -> int:
    api = GitHubAPI(
        end_point=cfg.end_point,
        num_discussion=cfg.num_discussion,
        num_comment=cfg.num_comment,
        num_reply=cfg.num_reply,
        min_credit=cfg.min_credit,
        out_dir=cfg.out_dir,
        dry_run=cfg.dry_run,
    )
    # Expect the implementation to provide a fetch_data() method as in the original script
    if not hasattr(api, "fetch_data"):
        raise AttributeError("GitHubAPI is missing a 'fetch_data()' method")
    api.fetch_data()

    return 0


def run_index(cfg: IndexGeneratorConfig) -> int:
    idx = IndexGenerator(
        load_local=cfg.load_local,
        model_path=cfg.model_path,
        model_name=cfg.model_name,
        show_progress=cfg.show_progress,
        rawdata=cfg.rawdata,
        database=cfg.database,
        dry_run=cfg.dry_run,
    )
    if not hasattr(idx, "generate_index"):
        raise AttributeError("IndexGenerator is missing a 'generate_index()' method")
    idx.generate_index()

    return 0


def run_bot(cfg: GitHubBotConfig) -> int:
    bot = GitHubBot(
        db_dir=cfg.db_dir,
        model_path=cfg.model_path,
        top_n=cfg.top_n,
        threshold=cfg.threshold,
        model_name=cfg.model_name,
        dry_run=cfg.dry_run,
        load_local=cfg.load_local,
    )
    if not hasattr(bot, "query_response"):
        raise AttributeError("GitHubBot is missing a 'query_response()' method")
    bot.query_response()

    return 0


def run_validation(cfg_api: GitHubAPIConfig, cfg_idx: IndexGeneratorConfig, cfg_bot: GitHubBotConfig, prompt: str | None = None,
                   pin_file: Path | None = None, golden_path: Path | None = None, write_golden: Path | None = None, fail_on_mismatch: bool = False) -> int:
    """
    Validation pipeline:
      1) Read pinned.txt (if provided) to determine exact discussions to fetch.
      2) Fetch those discussions via GitHubAPI into --val-out-dir.
      3) Build a vector DB from the fetched JSON into --val-db.
      4) Load the DB and answer the provided --prompt (offline retrieval).
      5) Optionally write a golden file and/or compare against an existing golden.
    """
    # Prepare output dirs
    out_dir = Path(cfg_api.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(cfg_idx.database)
    db_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Fetch data (pinned or generic) ---
    api = GitHubAPI(
        end_point=cfg_api.end_point,
        num_discussion=cfg_api.num_discussion,
        num_comment=cfg_api.num_comment,
        num_reply=cfg_api.num_reply,
        min_credit=cfg_api.min_credit,
        out_dir=str(out_dir),
        dry_run=cfg_api.dry_run,
    )
    if pin_file is not None and hasattr(api, "fetch_discussions_by_numbers"):
        owner, repo, numbers = utils.read_pin_file(pin_file)
        api.fetch_discussions_by_numbers(owner=owner, repo=repo, numbers=numbers, out_dir=str(out_dir))
    else:
        api.fetch_data()

    # --- Step 2: Build index from out_dir -> db_dir ---
    idx = IndexGenerator(
        load_local=cfg_idx.load_local,
        model_path=cfg_idx.model_path,
        model_name=cfg_idx.model_name,
        show_progress=cfg_idx.show_progress,
        rawdata=str(out_dir),
        database=str(db_dir),
        dry_run=cfg_idx.dry_run,
    )
    idx.generate_index()

    # --- Step 3: Load DB and answer prompt (offline) ---
    response_text = ""
    if prompt:
        bot = GitHubBot(
            db_dir=str(db_dir),
            model_path=cfg_bot.model_path,
            top_n=cfg_bot.top_n,
            threshold=cfg_bot.threshold,
            model_name=cfg_bot.model_name,
            dry_run=cfg_bot.dry_run,
            load_local=cfg_bot.load_local,
        )
        response_text = utils.call_bot_offline(bot, prompt, index=None, top_n=cfg_bot.top_n, threshold=cfg_bot.threshold, db_dir=db_dir)

    # --- Step 4: Golden write/compare ---
    if write_golden:
        Path(write_golden).write_text(response_text)
        print(f"Wrote golden → {write_golden}")

    if golden_path:
        expected = Path(golden_path).read_text()


        got_norm = utils.normalize_text_for_compare(response_text)
        exp_norm = utils.normalize_text_for_compare(expected)
        match = (got_norm == exp_norm)


        if not match:
            print("Validation mismatch against golden.")
            print("---- Diff ----")
            for line in difflib.unified_diff(
                expected.splitlines(), response_text.splitlines(), fromfile="golden", tofile="current", lineterm=""
            ):
                print(line)
            if fail_on_mismatch:
                return 1
        else:
            print("Validation matches golden.")

    # Always print the response for visibility
    if response_text:
        print(response_text)

    return 0


# -------------------------------
# Main
# -------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "debug", False):
        os.environ["DEBUG"] = "1"

    # Load YAML (if provided)
    cfg_from_yaml = AppConfig.from_yaml(args.config)

    # Apply CLI overrides to the active section
    cfg = cfg_from_yaml.override_with_cli(args, args.command)

    if args.print_config:
        to_dump: dict[str, Any]
        if args.command == "github-api":
            to_dump = {"github_api": asdict(cfg.github_api)}
        elif args.command == "index":
            to_dump = {"index": asdict(cfg.index)}
        elif args.command == "bot":
            to_dump = {"bot": asdict(cfg.bot)}
        else:
            to_dump = {}
        print(yaml.safe_dump(to_dump, sort_keys=False) if yaml else to_dump)
        return 0

    if args.command == "github-api":
        return run_github_api(cfg.github_api)
    if args.command == "index":
        return run_index(cfg.index)
    if args.command == "bot":
        return run_bot(cfg.bot)

    # validation: compose per-step configs based on overrides
    val_out_dir = getattr(args, "val_out_dir", "./validation/raw")
    val_db = getattr(args, "val_db", "./validation_db")
    prompt = getattr(args, "prompt", None)

    cfg_api = replace(cfg.github_api, out_dir=val_out_dir, dry_run=getattr(args, "dry_run", cfg.github_api.dry_run))
    cfg_idx = replace(
        cfg.index,
        rawdata=val_out_dir,
        database=val_db,
        model_path=getattr(args, "model_path", cfg.index.model_path) or cfg.index.model_path,
        model_name=getattr(args, "model_name", cfg.index.model_name) or cfg.index.model_name,
        load_local=cfg.index.load_local or getattr(args, "load_local", False),
        dry_run=getattr(args, "dry_run", cfg.index.dry_run),
    )
    cfg_bot = replace(
        cfg.bot,
        db_dir=val_db,
        model_path=getattr(args, "model_path", cfg.bot.model_path) or cfg.bot.model_path,
        model_name=getattr(args, "model_name", cfg.bot.model_name) or cfg.bot.model_name,
        load_local=cfg.bot.load_local or getattr(args, "load_local", False),
        dry_run=getattr(args, "dry_run", cfg.bot.dry_run),
    )

    return run_validation(
        cfg_api, cfg_idx, cfg_bot, prompt,
        pin_file=getattr(args, "pin_file", None),
        golden_path=getattr(args, "golden_path", None),
        write_golden=getattr(args, "write_golden", None),
        fail_on_mismatch=getattr(args, "fail_on_mismatch", False),
    )


if __name__ == "__main__":
    raise SystemExit(main())
