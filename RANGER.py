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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple
import io
import difflib
import re as _re

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

import inspect

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _normalize_text_for_compare(s: str) -> str:
    # Collapse whitespace and trim lines for robust comparisons
    lines = [line.strip() for line in s.strip().splitlines()]
    normalized = "\n".join(_re.sub(r"\s+", " ", ln) for ln in lines if ln != "")
    return normalized.strip()


def _read_pin_file(path: Path) -> Tuple[str | None, str | None, list[int]]:
    """
    Read lines like:
      owner/repo#123
      https://github.com/owner/repo/discussions/123
    Returns (owner, repo, [numbers]) when consistent; owner/repo may be None if not detectable.
    """
    owner = repo = None
    numbers: list[int] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http"):
            m = _re.fullmatch(r"https?://github\.com/([^/]+)/([^/]+)/discussions/(\d+)", line)
            if not m:
                raise ValueError(f"Unrecognized pin entry: {line}")
            o, r, n = m.group(1), m.group(2), int(m.group(3))
        else:
            m2 = _re.fullmatch(r"([^/]+)/([^#]+)#(\d+)", line)
            if not m2:
                raise ValueError(f"Unrecognized pin entry: {line}")
            o, r, n = m2.group(1), m2.group(2), int(m2.group(3))
        if owner is None and repo is None:
            owner, repo = o, r
        elif (o != owner) or (r != repo):
            # For simplicity we only support one repo in the pin file
            raise ValueError("Pin file must reference a single owner/repo.")
        numbers.append(n)
    if not numbers:
        raise ValueError("Pin file is empty or has no valid lines.")
    return owner, repo, numbers


# -----------------------------------------------------------------------------
# Import project modules
# -----------------------------------------------------------------------------

# Locate the project's src/ next to RANGER.py
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now you can import modules that live directly under src/
from ranger import GitHubAPI, IndexGenerator, GitHubBot


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
        if path is None and yaml is None:
            return cls()
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) if yaml else {}
        # Merge user YAML over defaults of a fresh instance
        merged = cls()
        merged_dict = asdict(merged)
        merged_dict = cls._deep_update(merged_dict, data or {})
        # Apply "default" to each section where the key is absent
        default_dict = asdict(merged.default)

        def section_with_defaults(section_data: Dict[str, Any], section_class):
            d = dict(default_dict)
            d.update(section_data or {})
            # Filter keys to those accepted by section_class
            field_names = {f.name for f in section_class.__dataclass_fields__.values()}
            return {k: v for k, v in d.items() if k in field_names}

        cfg = cls()
        # Use user-specified sections (falling back to default values)
        user_default = data.get("default", {}) if isinstance(data, dict) else {}
        if user_default:
            cfg.default = CommonConfig(**{k: v for k, v in user_default.items() if k in CommonConfig.__dataclass_fields__})
        user_api = data.get("github_api", {}) if isinstance(data, dict) else {}
        cfg.github_api = GitHubAPIConfig(**section_with_defaults(user_api, GitHubAPIConfig))
        user_idx = data.get("index", {}) if isinstance(data, dict) else {}
        cfg.index = IndexGeneratorConfig(**section_with_defaults(user_idx, IndexGeneratorConfig))
        user_bot = data.get("bot", {}) if isinstance(data, dict) else {}
        cfg.bot = GitHubBotConfig(**section_with_defaults(user_bot, GitHubBotConfig))
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
        # Apply overrides per command
        if command == "github-api":
            d = asdict(cfg.github_api)
            for k in d.keys():
                if getattr(args, k, None) is not None:
                    d[k] = getattr(args, k)
            cfg.github_api = GitHubAPIConfig(**d)
        elif command == "index":
            d = asdict(cfg.index)
            for k in d.keys():
                if getattr(args, k, None) is not None:
                    d[k] = getattr(args, k)
            cfg.index = IndexGeneratorConfig(**d)
        elif command == "bot":
            d = asdict(cfg.bot)
            for k in d.keys():
                if getattr(args, k, None) is not None:
                    d[k] = getattr(args, k)
            cfg.bot = GitHubBotConfig(**d)
        return cfg


# -----------------------------------------------------------------------------
# Bot call helper (offline‑first)
# -----------------------------------------------------------------------------

def _call_bot_and_capture(bot, prompt: str | None) -> str:
    """Return a response string without hitting GitHub.

    Preference order:
      1) If the bot exposes `generate_solution(prompt, top_n, index, threshold)`,
         call that directly using the loaded validation index.
      2) Otherwise, attempt to call `query_response(prompt?)` and capture stdout.
    """
    # --- Prefer an offline path if available (no network side-effects) ---
    if prompt and hasattr(bot, "generate_solution"):
        try:
            # Ensure the validation index is loaded
            if getattr(bot, "index", None) is None and hasattr(bot, "load_database"):
                _index = bot.load_database(getattr(bot, "db_dir", None))
                bot.index = _index
            top_n = getattr(bot, "top_n", 8)
            threshold = getattr(bot, "threshold", 0.0)
            text = bot.generate_solution(prompt, top_n, getattr(bot, "index", None), threshold)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            # Fall back to query_response path below
            pass

    # --- Fallback: call query_response and capture output ---
    result_text = None
    try:
        if hasattr(bot, "query_response"):
            sig = inspect.signature(bot.query_response)
            if len(sig.parameters) >= 1 and prompt is not None:
                ret = bot.query_response(prompt)
            else:
                ret = bot.query_response()
            if isinstance(ret, str):
                result_text = ret
    except Exception:
        # Fall through to stdout capture with a second call
        result_text = None

    if result_text is None and hasattr(bot, "query_response"):
        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            if prompt is not None:
                try:
                    bot.query_response(prompt)
                except TypeError:
                    bot.query_response()
            else:
                bot.query_response()
        finally:
            sys.stdout = old
        result_text = buf.getvalue()
    return (result_text or "").strip()


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

    # index
    pi = subs.add_parser("index", help="Generate a vector DB from raw GitHub JSON data.")
    pi.add_argument("--load-local", dest="load_local", action="store_true", help="Load a local model path.")
    pi.add_argument("--model-path", dest="model_path", type=str, help="Local model path.")
    pi.add_argument("--model-name", dest="model_name", type=str, help="Model name (HF ID or local dir).")
    pi.add_argument("--show-progress", dest="show_progress", action="store_true", help="Show progress bar.")
    pi.add_argument("--rawdata", dest="rawdata", type=str, help="Input data folder path.")
    pi.add_argument("--database", dest="database", type=str, help="Output index database path.")
    pi.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode.")

    # bot
    pb = subs.add_parser("bot", help="Run the bot against a vector DB.")
    pb.add_argument("--db-dir", dest="db_dir", type=str, help="Index data folder path.")
    pb.add_argument("--model-path", dest="model_path", type=str, help="Local model path.")
    pb.add_argument("--model-name", dest="model_name", type=str, help="Model name (HF ID or local dir).")
    pb.add_argument("--top-n", dest="top_n", type=int, help="The number of suggestions, range 1-10.")
    pb.add_argument("--threshold", dest="threshold", type=float, help="Relevance of suggestion, less than 1.0.")
    pb.add_argument("--dry-run", dest="dry_run", action="store_true", help="Dry run mode.")
    pb.add_argument("--load-local", dest="load_local", action="store_true", help="Load a local model path.")

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
    pv.add_argument("--compare", dest="compare_mode", choices=["normalize", "exact"], default="normalize", help="Comparison mode for golden check.")
    pv.add_argument("--fail-on-mismatch", dest="fail_on_mismatch", action="store_true", help="Exit non-zero if comparison fails.")

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
    if hasattr(api, "fetch_data"):
        api.fetch_data()
    else:
        raise AttributeError("GitHubAPI is missing a 'fetch_data()' method.")
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
    if hasattr(idx, "generate_index"):
        idx.generate_index()
    else:
        raise AttributeError("IndexGenerator is missing a 'generate_index()' method.")
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
    if hasattr(bot, "query_response"):
        bot.query_response()
    else:
        raise AttributeError("GitHubBot is missing a 'query_response()' method.")
    return 0


def run_validation(cfg_api: GitHubAPIConfig, cfg_idx: IndexGeneratorConfig, cfg_bot: GitHubBotConfig, prompt: str | None = None,
                   pin_file: Path | None = None, golden_path: Path | None = None, write_golden: Path | None = None,
                   compare_mode: str = "normalize", fail_on_mismatch: bool = False) -> int:
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
    if pin_file is not None:
        owner = repo = None
        try:
            owner, repo, numbers = _read_pin_file(pin_file)
        except Exception as e:
            raise RuntimeError(f"Failed to parse pin file {pin_file}: {e}")
        if hasattr(api, "fetch_discussions_by_numbers"):
            api.fetch_discussions_by_numbers(owner=owner, repo=repo, numbers=numbers, out_dir=str(out_dir))
        else:
            # Fallback to generic fetch if the specialized method is missing
            api.fetch_data()
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
    if hasattr(idx, "generate_index"):
        idx.generate_index()
    else:
        raise AttributeError("IndexGenerator is missing a 'generate_index()' method.")

    # --- Step 3: Load DB and answer prompt (offline) ---
    bot = GitHubBot(
        db_dir=str(db_dir),
        model_path=cfg_bot.model_path,
        top_n=cfg_bot.top_n,
        threshold=cfg_bot.threshold,
        model_name=cfg_bot.model_name,
        dry_run=cfg_bot.dry_run,
        load_local=cfg_bot.load_local,
    )
    response_text = ""
    if prompt:
        # Prefer offline retrieval API if available
        if hasattr(bot, "load_database"):
            index = bot.load_database(db_dir)
        else:
            index = None
        if hasattr(bot, "generate_solution"):
            response_text = bot.generate_solution(prompt, cfg_bot.top_n, index, cfg_bot.threshold)
        elif hasattr(bot, "query_response"):
            # Last resort
            maybe = bot.query_response(prompt)
            response_text = maybe if isinstance(maybe, str) else str(maybe)
    else:
        response_text = ""

    # --- Step 4: Golden write/compare ---
    if write_golden:
        Path(write_golden).write_text(response_text)
        print(f"Wrote golden → {write_golden}")

    if golden_path:
        try:
            expected = Path(golden_path).read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"Golden file not found: {golden_path}")

        if compare_mode == "normalize":
            got_norm = _normalize_text_for_compare(response_text)
            exp_norm = _normalize_text_for_compare(expected)
            match = (got_norm == exp_norm)
        else:
            match = (response_text == expected)

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

    # Load YAML (if provided)
    cfg_from_yaml = AppConfig.from_yaml(args.config)

    # Apply CLI overrides to the active section
    cfg = cfg_from_yaml.override_with_cli(args, args.command)

    if args.print_config:
        # Only print the config for the active subcommand
        if args.command == "github-api":
            print(yaml.safe_dump({"github_api": asdict(cfg.github_api)}, sort_keys=False) if yaml else asdict(cfg.github_api))
        elif args.command == "index":
            print(yaml.safe_dump({"index": asdict(cfg.index)}, sort_keys=False) if yaml else asdict(cfg.index))
        elif args.command == "bot":
            print(yaml.safe_dump({"bot": asdict(cfg.bot)}, sort_keys=False) if yaml else asdict(cfg.bot))
        elif args.command == "validation":
            # When invoked as a subcommand, set the same pipeline
            val_out_dir = getattr(args, "val_out_dir", "./validation/raw")
            val_db = getattr(args, "val_db", "./validation_db")
            prompt = getattr(args, "prompt", None)

            from dataclasses import replace
            cfg_api = replace(cfg.github_api, out_dir=val_out_dir)
            cfg_idx = replace(cfg.index, rawdata=val_out_dir, database=val_db)
            cfg_bot = replace(cfg.bot, db_dir=val_db)

            if getattr(args, "model_path", None):
                cfg_idx = replace(cfg_idx, model_path=args.model_path)
                cfg_bot = replace(cfg_bot, model_path=args.model_path)
            if getattr(args, "model_name", None):
                cfg_idx = replace(cfg_idx, model_name=args.model_name)
                cfg_bot = replace(cfg_bot, model_name=args.model_name)
            if getattr(args, "load_local", False):
                cfg_idx = replace(cfg_idx, load_local=True)
                cfg_bot = replace(cfg_bot, load_local=True)
            if getattr(args, "dry_run", False):
                cfg_api = replace(cfg_api, dry_run=True)
                cfg_idx = replace(cfg_idx, dry_run=True)
                cfg_bot = replace(cfg_bot, dry_run=True)

            return run_validation(
            cfg_api, cfg_idx, cfg_bot, prompt,
            pin_file=getattr(args, "pin_file", None),
            golden_path=getattr(args, "golden_path", None),
            write_golden=getattr(args, "write_golden", None),
            compare_mode=getattr(args, "compare_mode", "normalize"),
            fail_on_mismatch=getattr(args, "fail_on_mismatch", False),
        )

    # Dispatch
    if args.command == "github-api":
        return run_github_api(cfg.github_api)
    elif args.command == "index":
        return run_index(cfg.index)
    elif args.command == "bot":
        return run_bot(cfg.bot)
    elif args.command == "validation":
        # When invoked as a subcommand, set the same pipeline
        val_out_dir = getattr(args, "val_out_dir", "./validation/raw")
        val_db = getattr(args, "val_db", "./validation_db")
        prompt = getattr(args, "prompt", None)

        from dataclasses import replace
        cfg_api = replace(cfg.github_api, out_dir=val_out_dir)
        cfg_idx = replace(cfg.index, rawdata=val_out_dir, database=val_db)
        cfg_bot = replace(cfg.bot, db_dir=val_db)

        if getattr(args, "model_path", None):
            cfg_idx = replace(cfg_idx, model_path=args.model_path)
            cfg_bot = replace(cfg_bot, model_path=args.model_path)
        if getattr(args, "model_name", None):
            cfg_idx = replace(cfg_idx, model_name=args.model_name)
            cfg_bot = replace(cfg_bot, model_name=args.model_name)
        if getattr(args, "load_local", False):
            cfg_idx = replace(cfg_idx, load_local=True)
            cfg_bot = replace(cfg_bot, load_local=True)
        if getattr(args, "dry_run", False):
            cfg_api = replace(cfg_api, dry_run=True)
            cfg_idx = replace(cfg_idx, dry_run=True)
            cfg_bot = replace(cfg_bot, dry_run=True)

        return run_validation(
            cfg_api, cfg_idx, cfg_bot, prompt,
            pin_file=getattr(args, "pin_file", None),
            golden_path=getattr(args, "golden_path", None),
            write_golden=getattr(args, "write_golden", None),
            compare_mode=getattr(args, "compare_mode", "normalize"),
            fail_on_mismatch=getattr(args, "fail_on_mismatch", False),
        )

    # Shouldn't get here
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
