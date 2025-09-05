from __future__ import annotations

import io
import inspect
import re
from pathlib import Path
from typing import Tuple


def normalize_text_for_compare(s: str) -> str:
    """Normalize whitespace/lines to make text comparisons resilient."""
    lines = [line.strip() for line in (s or "").strip().splitlines()]
    normalized = "\n".join(re.sub(r"\s+", " ", ln) for ln in lines if ln)
    return normalized.strip()


def read_pin_file(path: Path) -> Tuple[str, str, list[int]]:
    """Parse pinned.txt lines like 'owner/repo#123' or full discussion URLs.

    Returns (owner, repo, [numbers]). Raises ValueError on mixed repos or bad lines.
    """
    owner = repo = None
    numbers: list[int] = []
    text = Path(path).read_text()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http"):
            m = re.fullmatch(r"https?://github\.com/([^/]+)/([^/]+)/discussions/(\d+)", line)
            if not m:
                raise ValueError(f"Unrecognized pin entry: {line}")
            o, r, n = m.group(1), m.group(2), int(m.group(3))
        else:
            m2 = re.fullmatch(r"([^/]+)/([^#]+)#(\d+)", line)
            if not m2:
                raise ValueError(f"Unrecognized pin entry: {line}")
            o, r, n = m2.group(1), m2.group(2), int(m2.group(3))
        if owner is None and repo is None:
            owner, repo = o, r
        elif (o != owner) or (r != repo):
            raise ValueError("Pin file must reference a single owner/repo.")
        numbers.append(n)
    if not numbers:
        raise ValueError("Pin file is empty or has no valid lines.")
    return owner, repo, numbers


def call_bot_offline(bot, prompt: str | None, *, index=None, top_n: int | None = None, threshold: float | None = None, db_dir: str | Path | None = None) -> str:
    """Return a response string without network side-effects.

    Prefers `generate_solution(prompt, top_n, index, threshold)` if available,
    otherwise falls back to calling `query_response` and capturing output.
    """
    if prompt and hasattr(bot, "generate_solution"):
        try:
            if getattr(bot, "index", None) is None and hasattr(bot, "load_database"):
                loaded = bot.load_database(getattr(bot, "db_dir", None))
                bot.index = loaded
            _top_n = top_n if top_n is not None else getattr(bot, "top_n", 8)
            _thr = threshold if threshold is not None else getattr(bot, "threshold", 0.0)
            text = bot.generate_solution(prompt, _top_n, getattr(bot, "index", None), _thr)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass  # fall through

    if db_dir is not None and not getattr(bot, "db_dir", None):
        bot.db_dir = str(db_dir)

    # Fallback: try to call query_response and capture.
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
        result_text = None

    if result_text is None and hasattr(bot, "query_response"):
        buf = io.StringIO()
        old = __import__("sys").stdout
        try:
            __import__("sys").stdout = buf
            if prompt is not None:
                try:
                    bot.query_response(prompt)
                except TypeError:
                    bot.query_response()
            else:
                bot.query_response()
        finally:
            __import__("sys").stdout = old
        result_text = buf.getvalue()

    return (result_text or "").strip()
