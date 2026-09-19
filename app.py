"""
Recallect — RAG-powered essay reviewer (Streamlit).

UI matches Recallect App UI.html (dark navy + logo cyan).
State machine: SETUP → GENERATION → INPUT → EVALUATION → REVEAL → LOOP
LLM: Groq (primary) with OpenRouter failover.
RAG: Exact concept match from data/RAG_Context.json (no vector DB).
Reviewer: Markdown notes from data/reviewer/Module_*.txt (one tab per module).
"""

from __future__ import annotations

import html
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
RAG_CONTEXT_PATH = APP_DIR / "data" / "RAG_Context.json"
REVIEWER_DIR = APP_DIR / "data" / "reviewer"
LOGO_PATH = APP_DIR / "assets" / "recollect_logo.png"

GROQ_MODEL = "llama-3.3-70b-versatile"
# If the preferred Llama id is unavailable on the account, try these Groq models next.
GROQ_FALLBACK_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "groq/compound",
    "openai/gpt-oss-20b",
)
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Brand palette (UI mock + logo cyan #4FBDE2)
BG = "#080C16"
ACCENT = "#4FBDE2"          # logo cyan (replaces mock #60A5FA)
ACCENT_SOFT = "#7DD3F0"     # lighter cyan (replaces #93C5FD)
ACCENT_BTN = "#2BA8D4"      # primary button (cyan family; mock used #3B82F6)
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
MUTED_2 = "#64748B"
BORDER = "rgba(79,189,226,0.30)"
BORDER_SOFT = "rgba(79,189,226,0.28)"

STAGE_SETUP = "SETUP"
STAGE_GENERATION = "GENERATION"
STAGE_INPUT = "INPUT"
STAGE_EVALUATION = "EVALUATION"
STAGE_REVEAL = "REVEAL"
STAGE_REVIEW = "REVIEW"

MODULE_META = {
    "Module 1: An Overview of Ethics": {
        "desc": (
            "What ethics, morals, and virtues are; corporate social responsibility; "
            "and how to bring ethical considerations into business and IT decisions."
        ),
        "icon": "scale",
        "reviewer": "Module_1.md",
    },
    "Module 2: Ethics for IT Workers and IT Users": {
        "desc": (
            "Relationships IT workers must manage, professional codes and piracy, "
            "and the ethical issues IT users face day to day."
        ),
        "icon": "laptop",
        "reviewer": "Module_2.md",
    },
    "Module 3: Cyberattacks and Cybersecurity": {
        "desc": (
            "Why incidents are so common, zero-day trade-offs, the CIA triad, "
            "and how organizations prevent and respond to cyberattacks."
        ),
        "icon": "shield",
        "reviewer": "Module_3.md",
    },
}

# Lucide-compatible path data (MIT) for inline SVG icons — no emoji.
_LUCIDE_PATHS: dict[str, str] = {
    "scale": (
        '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
        '<path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>'
    ),
    "laptop": (
        '<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>'
    ),
    "shield": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
    ),
    "sparkles": (
        '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>'
        '<path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>'
    ),
    "bot": (
        '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/>'
        '<path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>'
    ),
    "file-text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>'
    ),
    "pen-line": (
        '<path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/>'
    ),
    "circle-x": (
        '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>'
    ),
    "message-square": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
    "book-open": (
        '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>'
    ),
    "send": (
        '<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"/>'
        '<path d="m21.854 2.147-10.94 10.939"/>'
    ),
    "arrow-left": (
        '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>'
    ),
    "arrow-right": (
        '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>'
    ),
    "check": (
        '<path d="M20 6 9 17l-5-5"/>'
    ),
    "loader": (
        '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>'
    ),
    "library": (
        '<path d="m16 6 4 14"/><path d="M12 6v14"/><path d="M8 8v12"/><path d="M4 4v16"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
    ),
}


def icon(name: str, size: int = 20, color: str = ACCENT, stroke: float = 2) -> str:
    """Inline Lucide-style SVG icon."""
    paths = _LUCIDE_PATHS.get(name, _LUCIDE_PATHS["sparkles"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


# ---------------------------------------------------------------------------
# Secrets / environment
# ---------------------------------------------------------------------------

def get_secret(name: str) -> str | None:
    """Read an API key from Streamlit secrets, then fall back to env vars."""
    try:
        value = st.secrets.get(name)  # type: ignore[attr-defined]
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip() or None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_rag_context(_mtime: float, _size: int) -> dict[str, list[dict[str, Any]]]:
    if not RAG_CONTEXT_PATH.exists():
        raise FileNotFoundError(f"Missing RAG context file: {RAG_CONTEXT_PATH}")
    with RAG_CONTEXT_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError("RAG_Context.json must be a non-empty object of module arrays.")
    return data


def get_modules() -> dict[str, list[dict[str, Any]]]:
    if not RAG_CONTEXT_PATH.exists():
        return load_rag_context(0.0, 0)
    st_info = RAG_CONTEXT_PATH.stat()
    return load_rag_context(st_info.st_mtime, st_info.st_size)


def pick_random_concept(
    modules: dict[str, list[dict[str, Any]]],
    module_name: str,
    exclude_id: int | None = None,
) -> dict[str, Any]:
    concepts = modules.get(module_name) or []
    if not concepts:
        raise ValueError(f"No concepts found for module: {module_name}")
    pool = [c for c in concepts if c.get("id") != exclude_id] or concepts
    return random.choice(pool)


def _is_reviewer_heading(line: str) -> bool:
    """True for ALL-CAPS section titles (and numbered ones like '1. IT WORKERS…')."""
    s = line.strip()
    if len(s) < 3 or s.startswith("|") or s.startswith("-"):
        return False
    if s.startswith("#"):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.85


_HEADING_SMALL = {"a", "an", "and", "as", "at", "for", "in", "of", "on", "or", "the", "to", "vs", "vs."}
_HEADING_ACRONYMS = {
    "IT", "CSR", "FCC", "CIA", "BYOD", "DOS", "DDOS", "API", "IBM", "CEO", "HR", "AI",
    "IP", "SIIA", "BSA", "USA", "FBI", "IOS", "OS", "VPN", "USB", "PC", "ID", "QR",
    "AIG", "KFC", "ATM", "PII", "PHI", "GDPR", "ISO", "NIST", "APT", "SQL", "XSS",
}


def _title_case_heading(line: str) -> str:
    s = line.strip()
    # Preserve leading "1. " / "2. " numbering
    m = re.match(r"^(\d+\.\s+)(.*)$", s)
    prefix, body = (m.group(1), m.group(2)) if m else ("", s)
    words = body.split()
    out: list[str] = []
    for i, w in enumerate(words):
        trail = ":" if w.endswith(":") else ""
        core = w[:-1] if trail else w
        low = core.lower()
        up = core.upper()
        if up in _HEADING_ACRONYMS:
            out.append(up + trail)
        elif i > 0 and low in _HEADING_SMALL:
            out.append(low + trail)
        else:
            out.append(core[:1].upper() + core[1:].lower() + trail)
    return prefix + " ".join(out)


def reviewer_txt_to_markdown(raw: str) -> str:
    """Turn reviewer plain-text notes into markdown (headings + keep lists/tables)."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            out.append("")
            i += 1
            continue

        if re.match(r"^MODULE\s+\d+\s*$", stripped, re.I):
            title = stripped.title()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _is_reviewer_heading(lines[j]):
                out.append(f"# {title}: {_title_case_heading(lines[j])}")
                i = j + 1
                continue
            out.append(f"# {title}")
            i += 1
            continue

        if stripped.startswith("|") or stripped.startswith("-") or stripped.startswith("#"):
            out.append(lines[i].rstrip())
            i += 1
            continue

        if _is_reviewer_heading(stripped):
            out.append(f"## {_title_case_heading(stripped)}")
            i += 1
            continue

        out.append(lines[i].rstrip())
        i += 1

    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text + "\n"


@st.cache_data(show_spinner=False)
def load_reviewer_markdown(module_name: str, _mtime: float, _size: int) -> str:
    meta = MODULE_META.get(module_name) or {}
    filename = meta.get("reviewer") if isinstance(meta, dict) else None
    if not filename:
        raise FileNotFoundError(f"No reviewer file mapped for: {module_name}")
    path = REVIEWER_DIR / str(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing reviewer file: {path}")
    raw = path.read_text(encoding="utf-8")
    # .md files are already markdown; .txt notes are converted on the fly.
    if path.suffix.lower() == ".md":
        return raw if raw.endswith("\n") else raw + "\n"
    return reviewer_txt_to_markdown(raw)


def get_reviewer_markdown(module_name: str) -> str:
    meta = MODULE_META.get(module_name) or {}
    filename = meta.get("reviewer") if isinstance(meta, dict) else None
    path = REVIEWER_DIR / str(filename) if filename else None
    if path is None or not path.exists():
        return load_reviewer_markdown(module_name, 0.0, 0)
    st_info = path.stat()
    return load_reviewer_markdown(module_name, st_info.st_mtime, st_info.st_size)


def reviewer_markdown_to_html(md: str) -> str:
    """Render the limited markdown we generate (headings, lists, tables, paragraphs)."""
    lines = md.splitlines()
    parts: list[str] = []
    i = 0
    n = len(lines)

    def flush_paragraph(buf: list[str]) -> None:
        if not buf:
            return
        text = html.escape(" ".join(s.strip() for s in buf if s.strip()))
        parts.append(f"<p>{text}</p>")
        buf.clear()

    para: list[str] = []
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph(para)
            i += 1
            continue

        if stripped.startswith("# "):
            flush_paragraph(para)
            parts.append(f"<h1>{html.escape(stripped[2:].strip())}</h1>")
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph(para)
            parts.append(f"<h2>{html.escape(stripped[3:].strip())}</h2>")
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph(para)
            table_rows: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                # Skip markdown separator rows like |---|---|
                cells = [c.strip() for c in row.strip("|").split("|")]
                if cells and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
                    i += 1
                    continue
                table_rows.append(cells)
                i += 1
            if table_rows:
                parts.append("<table>")
                for r_idx, cells in enumerate(table_rows):
                    tag = "th" if r_idx == 0 else "td"
                    parts.append("<tr>")
                    for cell in cells:
                        parts.append(f"<{tag}>{html.escape(cell)}</{tag}>")
                    parts.append("</tr>")
                parts.append("</table>")
            continue

        if stripped.startswith("- "):
            flush_paragraph(para)
            parts.append("<ul>")
            while i < n and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:].strip()
                # continuation lines indented under the bullet
                i += 1
                while i < n and lines[i].startswith("  ") and not lines[i].strip().startswith("- "):
                    cont = lines[i].strip()
                    if cont:
                        item = f"{item} {cont}"
                    i += 1
                parts.append(f"<li>{html.escape(item)}</li>")
            parts.append("</ul>")
            continue

        para.append(stripped)
        i += 1

    flush_paragraph(para)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dual-provider LLM client (Groq → OpenRouter)
# ---------------------------------------------------------------------------

def _call_groq(system_prompt: str, user_prompt: str, api_key: str) -> str:
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None
    tried: list[str] = []
    for model in GROQ_FALLBACK_MODELS:
        if model in tried:
            continue
        tried.append(model)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=800,
            )
            message = completion.choices[0].message
            content = message.content
            if (not content or not str(content).strip()) and getattr(message, "reasoning", None):
                # Some Groq models put usable text only in reasoning; take the last non-empty line.
                lines = [ln.strip() for ln in str(message.reasoning).splitlines() if ln.strip()]
                content = lines[-1] if lines else None
            if not content or not str(content).strip():
                raise RuntimeError(f"Groq ({model}) returned an empty response.")
            if model != GROQ_MODEL:
                print(f"[Recallect] Groq using fallback model: {model}")
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[Recallect] Groq model {model} failed: {exc}")
            continue
    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


def _call_openrouter(system_prompt: str, user_prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://recollect.app",
        "X-Title": "Recallect Essay Reviewer",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 1200,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not content or not str(content).strip():
        raise RuntimeError("OpenRouter returned an empty response.")
    return str(content).strip()


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Try Groq first; on failure, fall back to OpenRouter."""
    groq_key = get_secret("GROQ_API_KEY")
    openrouter_key = get_secret("OPENROUTER_API_KEY")
    errors: list[str] = []

    if groq_key:
        try:
            return _call_groq(system_prompt, user_prompt, groq_key)
        except Exception as exc:  # noqa: BLE001
            warning = f"Groq failed ({exc}); falling back to OpenRouter."
            print(f"[Recallect] WARNING: {warning}")
            errors.append(warning)
    else:
        errors.append("GROQ_API_KEY is not set.")

    if openrouter_key:
        try:
            return _call_openrouter(system_prompt, user_prompt, openrouter_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"OpenRouter failed ({exc}).")
            print(f"[Recallect] ERROR: OpenRouter failed: {exc}")
    else:
        errors.append("OPENROUTER_API_KEY is not set.")

    raise RuntimeError("Both LLM providers failed or are unconfigured. " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Prompts & parsing
# ---------------------------------------------------------------------------

QUESTION_SYSTEM = (
    "You are an expert academic examiner. Your goal is to generate exactly ONE clear, "
    "concise, and focused essay question based ONLY on the provided context block. "
    "Do not make it overly complicated or multi-layered.\n\n"
    "To ensure the question is clear and targeted, you MUST think step-by-step using a "
    "Chain of Thought. Structure your response EXACTLY inside these markdown tags:\n\n"
    "<thinking>\n"
    "1. Identify the single most important concept or takeaway from the context.\n"
    "2. Formulate the core educational objective (What should the student prove they know?).\n"
    "3. Draft a simple phrasing that directly targets this objective without adding "
    "unnecessary fluff, jargon, or double-barreled clauses.\n"
    "4. Review the drafted question: Is it an essay format? Is it easy to understand? "
    "(Refine if it's too complicated).\n"
    "</thinking>\n\n"
    "<question>\n"
    "[Insert the finalized, single clear essay question here]\n"
    "</question>"
)

EVAL_SYSTEM = (
    "You are a strict academic grader for university essay answers. "
    "Evaluate the student's essay ONLY against the ground-truth context. "
    "Be specific and actionable. Do not invent facts absent from the context "
    "or essay. Use this exact markdown structure:\n\n"
    "## Score\n"
    "<integer 0-100>\n\n"
    "## Rating\n"
    "<Excellent | Good | Fair | Needs Improvement | Poor>\n\n"
    "## Summary\n"
    "<one sentence overall verdict>\n\n"
    "## Missing Details\n"
    "- <bullet for each key point from the ground truth that is weak or absent>\n\n"
    "## Constructive Feedback\n"
    "<2-4 sentences of concrete guidance>"
)


def build_question_prompt(concept: str, context: str) -> str:
    return (
        f"Concept: {concept}\n\n"
        f"Context block:\n{context}\n\n"
        "Follow the Chain of Thought format from the system prompt. "
        "Produce exactly one clear, focused essay question — not a multi-part prompt."
    )


def build_eval_prompt(concept: str, context: str, question: str, answer: str) -> str:
    return (
        f"Concept: {concept}\n\n"
        f"Essay question:\n{question}\n\n"
        f"Ground-truth context:\n{context}\n\n"
        f"Student essay:\n{answer}"
    )


def extract_question(llm_output: str) -> str:
    """Pull the finalized essay question from CoT <question> tags."""
    match = re.search(r"<question>(.*?)</question>", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def extract_thinking(llm_output: str) -> str | None:
    """Optional CoT reasoning block for terminal debug logging."""
    match = re.search(r"<thinking>(.*?)</thinking>", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _section(text: str, heading: str) -> str:
    pattern = rf"(?is)##\s*{re.escape(heading)}\s*\n+(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def parse_evaluation(evaluation: str) -> dict[str, Any]:
    score = None
    score_raw = _section(evaluation, "Score")
    score_match = re.search(r"(\d{1,3})", score_raw or evaluation)
    if score_match:
        score = max(0, min(100, int(score_match.group(1))))

    rating = _section(evaluation, "Rating").splitlines()[0].strip() if _section(evaluation, "Rating") else None
    if not rating:
        for label in ("Excellent", "Needs Improvement", "Good", "Fair", "Poor"):
            if re.search(rf"(?i)\b{re.escape(label)}\b", evaluation):
                rating = label
                break

    summary = _section(evaluation, "Summary")
    missing_block = _section(evaluation, "Missing Details")
    missing = [
        re.sub(r"^[-*•]\s*", "", line).strip()
        for line in missing_block.splitlines()
        if line.strip() and line.strip() not in ("-", "*", "•")
    ]
    feedback = _section(evaluation, "Constructive Feedback")
    return {
        "score": score,
        "rating": rating or "Graded",
        "summary": summary,
        "missing": missing,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults: dict[str, Any] = {
        "stage": STAGE_SETUP,
        "module_name": None,
        "concept": None,
        "question": None,
        "essay": "",
        "evaluation": None,
        "parsed": None,
        "last_concept_id": None,
        "error": None,
        "pending_generate": False,
        "pending_evaluate": False,
        "clicked_module": None,
        "review_module": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def open_reviewer(module_name: str | None = None) -> None:
    names = list(MODULE_META.keys())
    st.session_state.stage = STAGE_REVIEW
    st.session_state.review_module = module_name or names[0]
    st.session_state.error = None
    st.session_state.pending_generate = False
    st.session_state.pending_evaluate = False


def reset_question_state(keep_module: bool = True) -> None:
    module = st.session_state.get("module_name") if keep_module else None
    st.session_state.stage = STAGE_SETUP if not module else STAGE_GENERATION
    st.session_state.module_name = module
    st.session_state.concept = None
    st.session_state.question = None
    st.session_state.essay = ""
    st.session_state.evaluation = None
    st.session_state.parsed = None
    st.session_state.error = None
    st.session_state.pending_generate = False
    st.session_state.pending_evaluate = False
    if "essay_area" in st.session_state:
        del st.session_state["essay_area"]
    if not keep_module:
        st.session_state.clicked_module = None
        if "header_module_select" in st.session_state:
            del st.session_state["header_module_select"]


def start_module(module_name: str) -> None:
    st.session_state.module_name = module_name
    reset_question_state(keep_module=True)
    st.session_state.stage = STAGE_GENERATION
    st.session_state.pending_generate = True
    st.session_state.error = None
    st.session_state._gen_armed = False


# ---------------------------------------------------------------------------
# LLM stage runners
# ---------------------------------------------------------------------------

def run_generation(modules: dict[str, list[dict[str, Any]]]) -> None:
    module_name = st.session_state.module_name
    concept = pick_random_concept(
        modules, module_name, exclude_id=st.session_state.get("last_concept_id")
    )
    st.session_state.concept = concept
    st.session_state.last_concept_id = concept.get("id")
    st.session_state.error = None
    try:
        raw = call_llm(
            QUESTION_SYSTEM,
            build_question_prompt(concept["concept"], concept["context"]),
        )
        thinking = extract_thinking(raw)
        if thinking:
            print(f"[Recallect] Question CoT thinking:\n{thinking}\n")
        question = extract_question(raw)
        question = re.sub(r"^(Question\s*:?\s*)", "", question, flags=re.I).strip().strip('"')
        if not question:
            raise RuntimeError("LLM returned an empty essay question after CoT parsing.")
        st.session_state.question = question
        st.session_state.stage = STAGE_INPUT
        st.session_state.essay = ""
        st.session_state.evaluation = None
        st.session_state.parsed = None
        if "essay_area" in st.session_state:
            del st.session_state["essay_area"]
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = str(exc)
        st.session_state.stage = STAGE_SETUP
    finally:
        st.session_state.pending_generate = False


def run_evaluation() -> None:
    answer = (st.session_state.get("essay") or "").strip()
    concept = st.session_state.concept
    question = st.session_state.question
    st.session_state.error = None
    try:
        evaluation = call_llm(
            EVAL_SYSTEM,
            build_eval_prompt(concept["concept"], concept["context"], question, answer),
        )
        st.session_state.evaluation = evaluation
        st.session_state.parsed = parse_evaluation(evaluation)
        st.session_state.stage = STAGE_REVEAL
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = str(exc)
        st.session_state.stage = STAGE_INPUT
    finally:
        st.session_state.pending_evaluate = False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, div {{
  font-family: "IBM Plex Sans", system-ui, sans-serif !important;
}}

[data-testid="stAppViewContainer"] {{
  background: {BG};
  color: #E2E8F0;
  overflow: hidden !important;
  height: 100% !important;
  max-height: 100dvh !important;
}}
[data-testid="stHeader"],
header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  display: none !important;
}}
[data-testid="stToolbar"] {{
  display: none !important;
  height: 0 !important;
}}
html, body {{
  height: 100% !important;
  max-height: 100dvh !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
.stApp {{
  height: 100% !important;
  max-height: 100dvh !important;
  overflow: hidden !important;
}}
section.main, section[data-testid="stMain"], .stMain {{
  height: 100% !important;
  max-height: 100% !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
  padding: 0 !important;
  margin: 0 !important;
}}
/* Streamlit content-height resizer fights sticky layouts — disable it */
[data-testid="stAppIframeResizerAnchor"] {{
  display: none !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
}}
.block-container, [data-testid="stMainBlockContainer"],
.stMainBlockContainer.block-container {{
  position: relative !important;
  inset: auto !important;
  flex: 1 1 auto !important;
  padding-top: 1.15rem !important;
  padding-bottom: 1.15rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  max-width: 100% !important;
  width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
  margin: 0 !important;
}}
.block-container > div {{
  height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
}}
/* Root app column — Streamlit 1.3x+ may nest VB directly under block-container */
.block-container > [data-testid="stVerticalBlock"],
.block-container > div > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
  height: 100% !important;
  max-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  gap: 0.25rem !important;
  min-height: 0 !important;
}}

/* Kill Streamlit chrome completely (visibility:hidden still eats space) */
#MainMenu, footer, header {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}}
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
  display: none !important;
}}

/* Kill default Streamlit chrome gaps */
div[data-testid="stVerticalBlock"] > div:has(> div.rc-hide) {{ display: none; }}

.rc-shell {{ color: #E2E8F0; }}

.rc-topbar {{
  display: flex; align-items: center; justify-content: space-between; gap: 24px;
  padding: 22px 8px 18px 8px;
  border-bottom: 1px solid rgba(148,163,184,0.12);
  margin-bottom: 8px;
}}
.rc-brand {{ display: flex; align-items: center; gap: 10px; flex-shrink: 0; padding: 2px 0 4px 0; }}
.rc-logo {{
  width: 40px; height: 40px; object-fit: contain;
  display: block; background: transparent; border: none;
}}
.rc-brand-title {{ font-size: 22px; font-weight: 700; line-height: 1.1; }}
.rc-brand-title .re {{ color: {TEXT}; }}
.rc-brand-title .callect {{ color: {ACCENT}; }}
.rc-brand-tag {{ font-size: 12px; color: {MUTED}; }}

.rc-module-chip {{
  border: 1.5px solid rgba(79,189,226,0.45);
  background: rgba(43,168,212,0.06);
  border-radius: 16px; padding: 12px 16px;
  display: flex; align-items: stretch; gap: 14px;
  min-width: 340px; max-width: 520px; margin-left: auto;
}}
.rc-chip-shell {{
  border: 1.5px solid rgba(79,189,226,0.45);
  background: rgba(43,168,212,0.06);
  border-radius: 16px;
  padding: 12px 14px;
  display: block;
}}
.rc-chip-shell .ai-icon {{
  display: flex; align-items: center; justify-content: center;
  height: 100%; min-height: 52px;
}}
.rc-chip-row {{
  display: flex; align-items: center; gap: 10px; width: 100%;
}}
.rc-chip-row .select-slot {{ flex: 1; min-width: 0; }}
.rc-chip-row .status-slot {{ flex-shrink: 0; }}

.rc-status {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px; border-radius: 999px; flex-shrink: 0;
  font-size: 12px; font-weight: 500;
  white-space: nowrap;
  max-width: 100%;
  box-sizing: border-box;
}}
.rc-status-slot {{
  display: flex; justify-content: flex-end; align-items: center;
  min-height: 32px; width: 100%;
}}
.rc-status .dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
.rc-status.ready {{ background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #86EFAC; }}
.rc-status.ready .dot {{ background: #22C55E; }}
.rc-status.await {{ background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #FCD34D; }}
.rc-status.await .dot {{ background: #F59E0B; }}
.rc-status.busy {{ background: rgba(43,168,212,0.12); border: 1px solid rgba(79,189,226,0.35); color: {ACCENT_SOFT}; }}
.rc-status.done {{ background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #86EFAC; }}

/* Header module select — keep inside its column, no overflow onto status */
div[data-testid="stSelectbox"] {{ margin-bottom: 0 !important; max-width: 100%; }}
div[data-testid="stSelectbox"] label {{ display: none !important; }}
div[data-testid="stSelectbox"] > div {{ max-width: 100%; }}
div[data-testid="stSelectbox"] > div > div {{
  background: rgba(8,12,22,0.55) !important;
  border: 1.5px solid rgba(79,189,226,0.45) !important;
  border-radius: 10px !important;
  color: {TEXT} !important;
  min-height: 32px !important;
  overflow: hidden !important;
}}
/* Bordered header chip card */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border: 1.5px solid rgba(79,189,226,0.45) !important;
  background: rgba(43,168,212,0.06) !important;
  border-radius: 12px !important;
  padding: 4px 8px !important;
}}
/* Force dropdown | status to stay on one row inside the chip */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"] {{
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] {{
  min-width: 0 !important;
}}
/* Keep brand | chip header pair on one row on wide screens */
section.main .block-container > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type {{
  flex-direction: row !important;
  flex-wrap: nowrap !important;
}}

.rc-footer {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 4px 8px 4px;
  margin-top: 0;
  border-top: 1px solid rgba(148,163,184,0.12);
  background: {BG};
  box-sizing: border-box;
  min-height: 52px;
}}
.rc-footer .name {{ font-size: 14px; font-weight: 700; color: {ACCENT}; line-height: 1.3; }}
.rc-footer .copy {{ font-size: 11px; color: {MUTED_2}; line-height: 1.4; margin-top: 2px; }}
.rc-footer .right {{ text-align: right; }}
.rc-footer .right .t1 {{
  font-size: 12px; font-weight: 700; color: {TEXT};
  display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end;
  line-height: 1.3;
}}
.rc-footer .right .t2 {{ font-size: 11px; color: {ACCENT_SOFT}; line-height: 1.4; margin-top: 2px; }}

.rc-header-rule {{
  border-bottom: 1px solid rgba(148,163,184,0.12);
  margin: 2px 0 6px 0;
}}
.rc-scroll-marker {{ display: none !important; height: 0 !important; }}

/* Collapse inject_css / empty style markdown so it does not consume flex space */
.block-container [data-testid="stVerticalBlock"] > div:has(style),
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] > div:has(style) {{
  flex: 0 0 0 !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
}}

/* Pin header + footer; only the marked body region scrolls */
.block-container > [data-testid="stVerticalBlock"] > div:has(.rc-brand),
.block-container > div > [data-testid="stVerticalBlock"] > div:has(.rc-brand),
.block-container > [data-testid="stVerticalBlock"] > div:has(.rc-header-rule),
.block-container > div > [data-testid="stVerticalBlock"] > div:has(.rc-header-rule),
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > div:has(.rc-brand),
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > div:has(.rc-header-rule) {{
  flex: 0 0 auto !important;
  background: {BG} !important;
  z-index: 30 !important;
}}
.block-container > [data-testid="stVerticalBlock"] > div:has(.rc-footer),
.block-container > div > [data-testid="stVerticalBlock"] > div:has(.rc-footer),
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > div:has(.rc-footer) {{
  flex: 0 0 auto !important;
  margin-top: auto !important;
  background: {BG} !important;
  z-index: 30 !important;
}}
.block-container > [data-testid="stVerticalBlock"] > div:has(.rc-scroll-marker),
.block-container > div > [data-testid="stVerticalBlock"] > div:has(.rc-scroll-marker),
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > div:has(.rc-scroll-marker),
[data-testid="stLayoutWrapper"]:has(.rc-scroll-marker) {{
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}}
[data-testid="stElementContainer"]:has(.rc-footer),
[data-testid="stLayoutWrapper"]:has(.rc-footer) {{
  flex: 0 0 auto !important;
  margin-top: auto !important;
  overflow: visible !important;
  padding-bottom: 4px !important;
}}
[data-testid="stLayoutWrapper"]:has(.rc-brand),
[data-testid="stElementContainer"]:has(.rc-brand) {{
  flex: 0 0 auto !important;
  overflow: visible !important;
  padding-top: 2px !important;
}}
.block-container [data-testid="stVerticalBlock"] > div:has(.rc-scroll-marker) [data-testid="stVerticalBlock"] {{
  height: auto !important;
  max-height: none !important;
  overflow: visible !important;
}}

.rc-hero {{ text-align: center; margin: 18px 0 22px 0; }}
.rc-hero .title {{ font-size: 28px; font-weight: 700; color: {TEXT}; margin: 0 0 10px 0; }}
.rc-hero .sub {{ font-size: 15px; color: {MUTED}; margin: 0; }}

.rc-review-cta {{
  display: flex; align-items: center; justify-content: center; gap: 14px;
  margin: 0 0 8px 0;
}}
.rc-review-banner {{
  border: 1.5px solid rgba(79,189,226,0.35);
  background: rgba(43,168,212,0.07);
  border-radius: 16px;
  padding: 18px 22px;
  display: flex; align-items: center; justify-content: space-between; gap: 18px;
  margin: 0 0 20px 0;
  flex-wrap: wrap;
}}
.rc-review-banner .rb-left {{
  display: flex; align-items: flex-start; gap: 14px; min-width: 0; flex: 1;
}}
.rc-review-banner .rb-icon {{
  width: 42px; height: 42px; border-radius: 11px; flex-shrink: 0;
  background: rgba(79,189,226,0.14);
  display: flex; align-items: center; justify-content: center;
}}
.rc-review-banner .rb-title {{
  font-size: 17px; font-weight: 700; color: {TEXT}; margin: 0 0 4px 0;
}}
.rc-review-banner .rb-sub {{
  font-size: 13.5px; color: {MUTED}; margin: 0; line-height: 1.45;
}}

.rc-review-head {{
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 16px; margin: 4px 0 16px 0; flex-wrap: wrap;
}}
.rc-review-head .rh-title {{
  font-size: 24px; font-weight: 700; color: {TEXT}; margin: 0 0 6px 0;
}}
.rc-review-head .rh-sub {{
  font-size: 14px; color: {MUTED}; margin: 0; max-width: 640px; line-height: 1.45;
}}
.rc-review-meta {{
  font-size: 13px; color: {ACCENT_SOFT}; margin: 10px 0 14px 0;
}}
.rc-review-md {{
  font-size: 15px; line-height: 1.65; color: {TEXT};
  max-width: 820px;
}}
.rc-review-md h1 {{
  font-size: 26px; font-weight: 800; color: {TEXT};
  margin: 8px 0 12px 0; letter-spacing: -0.02em;
}}
.rc-review-md h2 {{
  font-size: 17px; font-weight: 700; color: {ACCENT_SOFT};
  margin: 22px 0 8px 0;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(79,189,226,0.22);
}}
.rc-review-md p {{
  margin: 0 0 10px 0; color: #CBD5E1;
}}
.rc-review-md ul, .rc-review-md ol {{
  margin: 0 0 12px 0; padding-left: 1.35em; color: #CBD5E1;
}}
.rc-review-md li {{ margin: 4px 0; }}
.rc-review-md table {{
  width: 100%; border-collapse: collapse; margin: 12px 0 16px 0;
  font-size: 13.5px;
}}
.rc-review-md th, .rc-review-md td {{
  border: 1px solid rgba(79,189,226,0.28);
  padding: 8px 10px; text-align: left; vertical-align: top;
}}
.rc-review-md th {{
  background: rgba(43,168,212,0.12); color: {ACCENT_SOFT}; font-weight: 700;
}}
.rc-review-md td {{ color: #CBD5E1; }}
.rc-topic-empty {{
  text-align: center; color: {MUTED}; padding: 36px 12px; font-size: 14.5px;
}}

/* Module cards: HTML face + invisible Streamlit button overlay */
.rc-mod-card {{
  border: 1.5px solid {BORDER_SOFT};
  border-radius: 18px;
  padding: 22px 22px 24px 22px;
  min-height: 210px;
  background: transparent;
  box-sizing: border-box;
  transition: border-color 0.15s ease, background 0.15s ease;
}}
.rc-mod-card.active {{
  border-color: {ACCENT_BTN};
  background: rgba(43,168,212,0.06);
}}
.rc-mod-card .rc-mod-icon {{
  width: 40px; height: 40px; border-radius: 10px;
  background: rgba(79,189,226,0.12);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 14px;
}}
.rc-mod-card .rc-mod-title {{
  font-size: 16px; font-weight: 700; color: {TEXT};
  margin: 0 0 10px 0; line-height: 1.35;
}}
.rc-mod-card .rc-mod-desc {{
  font-size: 13.5px; line-height: 1.55; color: {MUTED}; margin: 0;
}}

/* Column holds card + overlay button */
div[data-testid="stColumn"]:has(.rc-mod-card),
div[data-testid="column"]:has(.rc-mod-card),
div.stColumn:has(.rc-mod-card) {{
  position: relative !important;
}}
div[data-testid="stColumn"]:has(.rc-mod-card):hover .rc-mod-card,
div[data-testid="column"]:has(.rc-mod-card):hover .rc-mod-card,
div.stColumn:has(.rc-mod-card):hover .rc-mod-card {{
  border-color: {ACCENT};
  background: rgba(79,189,226,0.08);
}}
/* Invisible full-card click target */
div[data-testid="stColumn"]:has(.rc-mod-card) [class*="st-key-mod_"],
div[data-testid="column"]:has(.rc-mod-card) [class*="st-key-mod_"],
div.stColumn:has(.rc-mod-card) [class*="st-key-mod_"] {{
  position: absolute !important;
  inset: 0 !important;
  z-index: 6 !important;
  margin: 0 !important;
  height: 100% !important;
  width: 100% !important;
}}
div[data-testid="stColumn"]:has(.rc-mod-card) [class*="st-key-mod_"] button,
div[data-testid="column"]:has(.rc-mod-card) [class*="st-key-mod_"] button,
div.stColumn:has(.rc-mod-card) [class*="st-key-mod_"] button,
[class*="st-key-mod_"] button {{
  width: 100% !important;
  height: 100% !important;
  min-height: 210px !important;
  opacity: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  cursor: pointer !important;
  padding: 0 !important;
  color: transparent !important;
  font-size: 0 !important;
}}

.rc-panel {{
  border: 1.5px solid {BORDER};
  border-radius: 20px; padding: 26px 30px;
  margin: 12px 0;
}}
.rc-panel.dim {{ opacity: 0.45; }}
.rc-panel-title {{
  display: flex; align-items: center; gap: 10px;
  font-size: 17px; font-weight: 700; color: {TEXT}; margin-bottom: 16px;
}}
.rc-qbox {{
  border: 1.5px solid rgba(79,189,226,0.35);
  border-radius: 14px; padding: 22px 26px;
  font-size: 17px; line-height: 1.55; color: {TEXT}; font-weight: 500;
}}

.rc-score-wrap {{
  border: 1.5px solid {BORDER}; border-radius: 20px; padding: 24px 30px;
  display: flex; align-items: center; gap: 26px; margin: 12px 0;
}}
.rc-score-ring {{
  width: 84px; height: 84px; border-radius: 50%;
  border: 5px solid #22C55E;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}}
.rc-score-ring .n {{ font-size: 27px; font-weight: 700; color: {TEXT}; }}
.rc-score-ring .d {{ font-size: 12px; color: {MUTED_2}; }}
.rc-rating-pill {{
  padding: 4px 14px; background: rgba(34,197,94,0.14);
  border: 1px solid rgba(34,197,94,0.3); color: #86EFAC;
  border-radius: 999px; font-size: 12.5px; font-weight: 600;
}}
.rc-miss {{
  font-size: 13.5px; line-height: 1.55; color: #CBD5E1;
  padding-left: 14px; border-left: 2px solid rgba(248,113,113,0.4);
  margin: 8px 0;
}}
.rc-gt {{
  border: 1.5px solid rgba(129,140,248,0.4); border-radius: 20px;
  padding: 22px 30px; margin: 12px 0;
}}
.rc-gt .label {{ font-size: 13px; font-weight: 700; color: #818CF8; margin-bottom: 10px; }}
.rc-gt .body {{ font-size: 14.5px; line-height: 1.65; color: #CBD5E1; }}

.rc-center {{
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 24px; min-height: 420px; text-align: center; padding: 40px 16px;
}}
.rc-spinner {{
  width: 96px; height: 96px; border-radius: 50%;
  border: 5px solid rgba(79,189,226,0.18);
  border-top-color: {ACCENT};
  animation: rc-spin 0.9s linear infinite;
}}
.rc-overlay-card {{
  background: #0F1729; border: 1.5px solid rgba(79,189,226,0.4);
  border-radius: 20px; padding: 36px 48px; text-align: center;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5); max-width: 420px; margin: 0 auto;
}}
.rc-pulse {{
  height: 16px; border-radius: 8px; background: rgba(148,163,184,0.14);
  animation: rc-pulse 1.4s ease-in-out infinite;
}}
@keyframes rc-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes rc-pulse {{
  0%, 100% {{ opacity: 0.45; }}
  50% {{ opacity: 1; }}
}}

/* Buttons */
div[data-testid="stButton"] > button {{
  border-radius: 12px !important;
  font-weight: 700 !important;
  font-family: "IBM Plex Sans", system-ui, sans-serif !important;
}}
div[data-testid="stButton"] > button[kind="primary"] {{
  background: {ACCENT_BTN} !important;
  border-color: {ACCENT_BTN} !important;
  color: #fff !important;
}}
div[data-testid="stButton"] > button[kind="secondary"] {{
  background: transparent !important;
  border: 1.5px solid {BORDER_SOFT} !important;
  color: {ACCENT} !important;
}}

/* Text area */
div[data-testid="stTextArea"] textarea {{
  background: transparent !important;
  color: {TEXT} !important;
  border: 1.5px solid rgba(79,189,226,0.35) !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  line-height: 1.6 !important;
  min-height: 160px !important;
}}

/* Hide default label excess */
.stCaption {{ color: {MUTED_2} !important; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def logo_data_uri() -> str:
    """Embed logo as data URI so custom HTML can show it."""
    import base64

    if not LOGO_PATH.exists():
        return ""
    raw = LOGO_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def status_html(stage: str) -> str:
    """Compact status pill for the header chip (stays on one line beside the dropdown)."""
    if stage in (STAGE_SETUP, STAGE_REVIEW):
        return '<div class="rc-status ready"><div class="dot"></div>Ready</div>'
    if stage == STAGE_GENERATION:
        return (
            f'<div class="rc-status busy">{icon("loader", 13, ACCENT_SOFT, 2.6)} '
            f"Generating…</div>"
        )
    if stage == STAGE_INPUT:
        return '<div class="rc-status await"><div class="dot"></div>Awaiting…</div>'
    if stage == STAGE_EVALUATION:
        return (
            f'<div class="rc-status busy">{icon("loader", 13, ACCENT_SOFT, 2.6)} '
            f"Evaluating…</div>"
        )
    return (
        f'<div class="rc-status done">{icon("check", 13, "#86EFAC", 2.6)} Complete</div>'
    )


def _brand_block(logo: str) -> None:
    st.markdown(
        f"""
        <div class="rc-brand" style="padding: 2px 0;">
          {"<img class='rc-logo' src='" + logo + "' alt='Recallect logo' />" if logo else ""}
          <div>
            <div class="rc-brand-title"><span class="re">Recall</span><span class="callect">ect</span></div>
            <div class="rc-brand-tag">Study Smarter. Recall Better.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(
    stage: str,
    module_names: list[str] | None = None,
    show_module_controls: bool = True,
) -> None:
    """Brand left. Landing: Ready only. Workflow pages: compact chip with dropdown | status."""
    logo = logo_data_uri()
    names = module_names or list(MODULE_META.keys())
    current = st.session_state.get("module_name")

    if not show_module_controls:
        left, right = st.columns([3, 1], gap="large")
        with left:
            _brand_block(logo)
        with right:
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;align-items:center;min-height:48px;">{status_html(stage)}</div>',
                unsafe_allow_html=True,
            )
    else:
        left, right = st.columns([1.05, 1.15], gap="medium")
        with left:
            _brand_block(logo)

        with right:
            with st.container(border=True):
                # Compact one row: AI icon | dropdown | status (no "Current Module" label)
                ic_col, sel_col, status_col = st.columns([0.09, 0.68, 0.23], gap="small")
                with ic_col:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:center;min-height:36px;">{icon("bot", 30, ACCENT, 1.7)}</div>',
                        unsafe_allow_html=True,
                    )
                with sel_col:
                    index = names.index(current) if current in names else 0
                    selected = st.selectbox(
                        "Current Module",
                        options=names,
                        index=index,
                        label_visibility="collapsed",
                        key="header_module_select",
                    )
                with status_col:
                    st.markdown(
                        f'<div class="rc-status-slot">{status_html(stage)}</div>',
                        unsafe_allow_html=True,
                    )

            if selected != current:
                start_module(selected)
                st.rerun()

    st.markdown('<div class="rc-header-rule"></div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="rc-footer">
          <div>
            <div class="name">Recallect</div>
            <div class="copy">Copyright 2026</div>
          </div>
          <div class="right">
            <div class="t1">{icon("sparkles", 16, "#A78BFA")} RAG-Powered Essay Reviewer</div>
            <div class="t2">Powered by Groq &amp; OpenRouter</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scroll_body():
    """Scrollable middle region between sticky header and footer."""
    box = st.container()
    box.markdown('<div class="rc-scroll-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
    if st.session_state.get("error"):
        box.error(st.session_state.error)
    return box


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def screen_setup(modules: dict[str, list[dict[str, Any]]]) -> None:
    names = list(modules.keys())

    render_topbar(STAGE_SETUP, module_names=names, show_module_controls=False)
    with scroll_body():
        ban_l, ban_r = st.columns([3.2, 1.1], gap="medium")
        with ban_l:
            st.markdown(
                f"""
                <div class="rc-review-banner" style="margin:0;">
                  <div class="rb-left">
                    <div class="rb-icon">{icon("library", 22, ACCENT)}</div>
                    <div>
                      <div class="rb-title">Module Summary Reviewer</div>
                      <div class="rb-sub">
                        Read every topic and a clear description first — then practice essays when you are ready.
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ban_r:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            if st.button("Open Reviewer", type="primary", use_container_width=True, key="open_reviewer"):
                open_reviewer(names[0])
                st.rerun()

        st.markdown(
            """
            <div class="rc-hero">
              <div class="title">Or jump straight into essay practice</div>
              <div class="sub">Tap a module and Recallect immediately writes your first question — no extra steps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clicked = st.session_state.get("clicked_module")
        cols = st.columns(3, gap="medium")
        for i, (col, name) in enumerate(zip(cols, names)):
            meta = MODULE_META.get(name, {})
            desc = meta.get("desc", "Explore concepts from this module.") if isinstance(meta, dict) else str(meta)
            icon_name = meta.get("icon", "book-open") if isinstance(meta, dict) else "book-open"
            active = clicked == name
            active_cls = " active" if active else ""
            with col:
                st.markdown(
                    f"""
                    <div class="rc-mod-card{active_cls}">
                      <div class="rc-mod-icon">{icon(icon_name, 19, ACCENT)}</div>
                      <div class="rc-mod-title">{html.escape(name)}</div>
                      <div class="rc-mod-desc">{html.escape(desc)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Invisible overlay button — makes the whole card clickable
                if st.button(
                    f"Open {name}",
                    key=f"mod_{i}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.clicked_module = name
                    start_module(name)
                    st.rerun()

    render_footer()


def screen_review(modules: dict[str, list[dict[str, Any]]]) -> None:
    names = list(modules.keys())
    current = st.session_state.get("review_module") or names[0]
    if current not in names:
        current = names[0]
        st.session_state.review_module = current

    render_topbar(STAGE_REVIEW, module_names=names, show_module_controls=False)
    with scroll_body():
        meta = MODULE_META.get(current, {})
        mod_desc = meta.get("desc", "") if isinstance(meta, dict) else ""
        icon_name = meta.get("icon", "book-open") if isinstance(meta, dict) else "book-open"

        st.markdown(
            f"""
            <div class="rc-review-head">
              <div>
                <div class="rh-title" style="display:flex;align-items:center;gap:10px;">
                  {icon("library", 26, ACCENT)} Module Summary Reviewer
                </div>
                <div class="rh-sub">
                  Study the module notes as markdown before essay practice — one tab per module.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_l, nav_r = st.columns([1, 1])
        with nav_l:
            if st.button("← Back to home", use_container_width=True, key="review_back"):
                st.session_state.stage = STAGE_SETUP
                st.rerun()
        with nav_r:
            if st.button("Practice essays on this module →", type="primary", use_container_width=True, key="review_practice"):
                st.session_state.clicked_module = current
                start_module(current)
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        tab_cols = st.columns(len(names), gap="small")
        for i, (col, name) in enumerate(zip(tab_cols, names)):
            with col:
                selected = name == current
                short = re.sub(r"^(Module\s+\d+).*", r"\1", name)
                if st.button(
                    short,
                    key=f"review_tab_{i}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.review_module = name
                    st.rerun()

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin:14px 0 4px 0;">
              <div class="rc-mod-icon" style="width:36px;height:36px;margin:0;">{icon(icon_name, 18, ACCENT)}</div>
              <div>
                <div style="font-size:18px;font-weight:700;color:{TEXT};">{html.escape(current)}</div>
                <div style="font-size:13px;color:{MUTED};">{html.escape(mod_desc)}</div>
              </div>
            </div>
            <div class="rc-review-meta">Study notes · full module on one page</div>
            """,
            unsafe_allow_html=True,
        )

        try:
            md = get_reviewer_markdown(current)
            body_html = reviewer_markdown_to_html(md)
            st.markdown(
                f'<div class="rc-review-md">{body_html}</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f'<div class="rc-topic-empty">Could not load reviewer notes: {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button(
            f"Done reviewing — start essay practice ({current})",
            type="primary",
            use_container_width=True,
            key="review_practice_bottom",
        ):
            st.session_state.clicked_module = current
            start_module(current)
            st.rerun()

    render_footer()


def screen_generating() -> None:
    render_topbar(STAGE_GENERATION, module_names=list(MODULE_META.keys()))
    module = st.session_state.module_name or ""
    with scroll_body():
        st.markdown(
            f"""
            <div class="rc-center">
              <div class="rc-spinner"></div>
              <div>
                <div style="font-size:21px;font-weight:700;color:{TEXT};margin-bottom:8px;">
                  Generating your essay question
                </div>
                <div style="font-size:14.5px;color:{MUTED};">
                  Pulling a concept from {html.escape(module)} and asking the model to write a question…
                </div>
              </div>
              <div style="width:100%;max-width:720px;display:flex;flex-direction:column;gap:12px;">
                <div class="rc-pulse" style="width:40%;"></div>
                <div class="rc-pulse" style="width:100%;"></div>
                <div class="rc-pulse" style="width:88%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_footer()


def screen_input() -> None:
    render_topbar(STAGE_INPUT, module_names=list(MODULE_META.keys()))
    question = st.session_state.question or ""

    with scroll_body():
        st.markdown(
            f"""
            <div class="rc-panel">
              <div class="rc-panel-title">{icon("file-text", 19, TEXT)} Essay Question</div>
              <div class="rc-qbox">{html.escape(question)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="rc-panel"><div class="rc-panel-title">{icon("pen-line", 17, TEXT)} Your Essay Answer</div>',
            unsafe_allow_html=True,
        )
        if "essay_area" not in st.session_state:
            st.session_state.essay_area = st.session_state.essay or ""
        st.text_area(
            "Essay answer",
            height=180,
            max_chars=2000,
            placeholder="Type your essay answer here...",
            label_visibility="collapsed",
            key="essay_area",
        )
        st.session_state.essay = st.session_state.essay_area
        char_count = len(st.session_state.essay or "")

        left, right = st.columns([1, 1])
        with left:
            st.caption(f"{char_count} / 2000")
        with right:
            submit_label = "Submit Answer"
            if st.button(submit_label, type="primary", use_container_width=True):
                answer = (st.session_state.get("essay_area") or "").strip()
                st.session_state.essay = answer
                if not answer:
                    st.warning("Please write an essay answer before submitting.")
                elif len(answer) < 40:
                    st.warning("Your answer is too short. Aim for a few thoughtful sentences.")
                else:
                    st.session_state.stage = STAGE_EVALUATION
                    st.session_state.pending_evaluate = True
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    render_footer()


def screen_evaluating() -> None:
    render_topbar(STAGE_EVALUATION, module_names=list(MODULE_META.keys()))
    question = html.escape(st.session_state.question or "")
    answer = html.escape(st.session_state.essay or "")
    with scroll_body():
        st.markdown(
            f"""
            <div style="position:relative;">
              <div class="rc-panel dim">
                <div class="rc-panel-title">{icon("file-text", 19, TEXT)} Essay Question</div>
                <div class="rc-qbox" style="font-size:16px;color:#CBD5E1;">{question}</div>
              </div>
              <div class="rc-panel dim">
                <div class="rc-panel-title">{icon("pen-line", 17, TEXT)} Your Essay Answer</div>
                <div class="rc-qbox" style="font-size:15px;color:#CBD5E1;font-weight:400;">{answer}</div>
              </div>
              <div class="rc-overlay-card" style="margin-top:8px;">
                <div class="rc-spinner" style="width:64px;height:64px;margin:0 auto 18px;"></div>
                <div style="font-size:18px;font-weight:700;color:{TEXT};margin-bottom:6px;">
                  Evaluating your response
                </div>
                <div style="font-size:13.5px;color:{MUTED};">
                  Comparing your answer against the ground-truth context to score it and write feedback — this happens automatically.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_footer()


def screen_reveal() -> None:
    render_topbar(STAGE_REVEAL, module_names=list(MODULE_META.keys()))
    parsed = st.session_state.parsed or parse_evaluation(st.session_state.evaluation or "")
    concept = st.session_state.concept or {}
    score = parsed.get("score")
    rating = parsed.get("rating") or "Graded"
    summary = parsed.get("summary") or ""
    missing = parsed.get("missing") or []
    feedback = parsed.get("feedback") or ""
    module = st.session_state.module_name or ""
    concept_name = concept.get("concept", "")

    score_html = (
        f'<span class="n">{score}</span><span class="d">/100</span>'
        if score is not None
        else '<span class="n">—</span>'
    )
    miss_html = "".join(
        f'<div class="rc-miss">{html.escape(item)}</div>' for item in missing
    ) or '<div class="rc-miss">No major gaps detected against the ground truth.</div>'

    with scroll_body():
        st.markdown(
            f"""
            <div class="rc-score-wrap">
              <div class="rc-score-ring">{score_html}</div>
              <div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                  <span class="rc-rating-pill">{html.escape(rating)}</span>
                  <span style="font-size:13px;color:{MUTED_2};">{html.escape(module)} · {html.escape(concept_name)}</span>
                </div>
                <div style="font-size:14.5px;line-height:1.5;color:#CBD5E1;">{html.escape(summary)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                f"""
                <div class="rc-panel">
                  <div class="rc-panel-title">{icon("circle-x", 17, "#F87171")} What You Missed</div>
                  {miss_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="rc-panel">
                  <div class="rc-panel-title">{icon("message-square", 17, ACCENT)} Constructive Feedback</div>
                  <div style="font-size:13.5px;line-height:1.6;color:#CBD5E1;">{html.escape(feedback)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="rc-gt">
              <div class="label" style="display:flex;align-items:center;gap:9px;">
                {icon("book-open", 17, "#818CF8")} Ground Truth · {html.escape(module)}
              </div>
              <div class="body">{html.escape(concept.get("context", ""))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])
        with left:
            if st.button("Back to module", use_container_width=True):
                st.session_state.clicked_module = None
                reset_question_state(keep_module=False)
                st.rerun()
        with right:
            if st.button("Next Question", type="primary", use_container_width=True):
                last_id = st.session_state.get("last_concept_id")
                reset_question_state(keep_module=True)
                st.session_state.last_concept_id = last_id
                st.session_state.stage = STAGE_GENERATION
                st.session_state.pending_generate = True
                st.rerun()
            st.caption(f"Automatically generates a new question from {module}")

    render_footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Recallect",
        page_icon="assets/recollect_logo.png" if LOGO_PATH.exists() else None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    init_state()

    try:
        modules = get_modules()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load module data: {exc}")
        st.stop()

    # Pending async-style stage transitions (keep spinner screen visible one frame)
    if st.session_state.pending_generate and st.session_state.stage == STAGE_GENERATION:
        # Show generating UI first, then run on this same pass after paint isn't possible —
        # so: if we just entered, show spinner; if spinner already shown via flag, run.
        if st.session_state.get("_gen_armed"):
            run_generation(modules)
            st.session_state._gen_armed = False
            st.rerun()
        else:
            st.session_state._gen_armed = True
            screen_generating()
            st.rerun()
        return

    if st.session_state.pending_evaluate and st.session_state.stage == STAGE_EVALUATION:
        if st.session_state.get("_eval_armed"):
            run_evaluation()
            st.session_state._eval_armed = False
            st.rerun()
        else:
            st.session_state._eval_armed = True
            screen_evaluating()
            st.rerun()
        return

    stage = st.session_state.stage
    if stage == STAGE_SETUP:
        screen_setup(modules)
    elif stage == STAGE_REVIEW:
        screen_review(modules)
    elif stage == STAGE_GENERATION:
        screen_generating()
    elif stage == STAGE_INPUT:
        screen_input()
    elif stage == STAGE_EVALUATION:
        screen_evaluating()
    elif stage == STAGE_REVEAL:
        screen_reveal()
    else:
        screen_setup(modules)


if __name__ == "__main__":
    main()
