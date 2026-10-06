#!/usr/bin/env python3
"""
SolveMate AI  -  Advanced Telegram Bot powered by Google Gemini

Features
--------
* Real chat history (proper user/model roles) saved to disk
* Live streaming answers (message updates while Gemini is typing)
* Pretty Telegram formatting (bold, code blocks, lists, links)
* Google Search grounding (up-to-date answers + sources)
* Understands photos, PDFs, text/code files, voice messages
* Expert modes: General, Coder, Teacher, Math, Career, Writer
* Automatic model fallback + retry with backoff
* Multi-user safe (concurrent updates, per-chat lock, rate limit)
* Inline menu, /stop /resume /clear /mode /search /id /stats

Install:
    pip install -U python-telegram-bot google-genai

Run (recommended: use environment variables):
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    export GEMINI_API_KEY="AIza..."
    python solvemate_bot.py
"""

import asyncio
import html
import json
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models are tried in order. If the first one fails/overloaded, the next one is used.
# Change these to any valid Gemini model name from Google AI Studio.
# Example for best quality:  GEMINI_MODELS="gemini-2.5-pro,gemini-2.5-flash"
MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-1.5-flash").split(",")
    if m.strip()
]

ADMIN_IDS = {
    int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()
}

ENABLE_SEARCH = True               # Google Search grounding (users can toggle)
MAX_HISTORY_MESSAGES = 24          # messages (user + model) remembered per chat
MAX_RETRIES = 3                    # retries per model for temporary errors
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.7
REQUESTS_PER_MINUTE = 10           # per user
MAX_FILE_MB = 15
STREAM_INTERVAL = 1.4              # seconds between live edits (avoid Telegram flood)
REQUEST_TIMEOUT_MS = 120_000
DATA_FILE = Path(os.getenv("DATA_FILE", "solvemate_data.json"))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("solvemate")


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
)


# ============================================================
# 3. MODES (EXPERT PERSONAS)
# ============================================================

MODES = {
    "general": {
        "label": "🧠 General",
        "prompt": "",
    },
    "coder": {
        "label": "💻 Coder",
        "prompt": (
            "MODE: Senior Software Engineer.\n"
            "- Find the root cause of bugs and explain it clearly.\n"
            "- Always give complete, working, copy-paste-ready code in fenced code blocks "
            "with the language name.\n"
            "- Mention which file the code goes in and how to run/test it.\n"
            "- Point out edge cases, security issues and better practices briefly.\n"
            "- Prefer modern, idiomatic code. Never give half-finished snippets with '...'."
        ),
    },
    "teacher": {
        "label": "📚 Teacher",
        "prompt": (
            "MODE: Expert Teacher.\n"
            "- Start with a simple definition, then explain the idea step by step.\n"
            "- Use an easy real-life analogy and at least one worked example.\n"
            "- Finish with a short summary or key points for exam revision.\n"
            "- Adjust the depth to the student's level."
        ),
    },
    "math": {
        "label": "🧮 Math",
        "prompt": (
            "MODE: Math & Logic Solver.\n"
            "- Solve step by step, showing every calculation.\n"
            "- Double-check the final answer and state it clearly at the end.\n"
            "- Use plain text math (like x^2, sqrt(x), a/b) because Telegram cannot render LaTeX."
        ),
    },
    "career": {
        "label": "💼 Career",
        "prompt": (
            "MODE: Career Coach & Recruiter.\n"
            "- Give practical, honest, specific advice for resumes, interviews and job search.\n"
            "- When writing resumes or emails, produce a ready-to-use final version.\n"
            "- For interview prep, give likely questions with strong sample answers."
        ),
    },
    "writer": {
        "label": "✍️ Writer",
        "prompt": (
            "MODE: Professional Writer & Editor.\n"
            "- Produce polished, natural, well-structured writing in the requested tone.\n"
            "- When editing, keep the user's meaning and briefly say what you improved.\n"
            "- Offer a shorter or more formal variant only when it is useful."
        ),
    },
}

BASE_SYSTEM_INSTRUCTION = """
You are SolveMate AI, a smart, accurate and friendly general-purpose AI assistant inside Telegram.
You help with programming, studies, mathematics, career, projects, technology and everyday problems.

QUALITY RULES
1. Understand the user's real goal first, then answer completely and correctly.
   Think carefully before answering; for hard problems reason step by step internally.
2. Reply in the SAME language and script the user writes in. If the user writes Telugu, Hindi or any
   language in English letters (for example Telugu typed in English), reply in that same style.
3. Be direct. Give the answer first, then the explanation. No filler, no repeating the question.
4. Coding: find the error, explain why it happens, give full corrected code, explain the fix.
   Never leave placeholders like "your code here" when you can write the real code.
5. Studies: definition -> simple explanation -> example -> short summary.
6. Math: show calculations step by step and verify the result.
7. For projects: say which file to create, where code goes, and how to run it.
8. If something is uncertain or you do not know, say so honestly. Never invent facts, links,
   libraries, functions or quotes. Do not claim you performed actions you did not perform.
9. If the request is truly ambiguous, ask ONE short clarifying question; otherwise make a sensible
   assumption, state it briefly and continue.
10. If the user sends an image, file or voice message, analyse it carefully and answer about it.

TELEGRAM FORMATTING RULES
- Use **bold** for key terms, `inline code` for code words, and fenced code blocks with a language
  name for code.
- Use short paragraphs and simple "-" bullet lists. Avoid very long headings; do not use LaTeX.
- Avoid wide tables (Telegram screens are narrow). Use lists instead.
- Keep normal answers focused; go long only when the task needs it (code, essays, deep explanations).
""".strip()


def build_system_instruction(mode_key: str) -> str:
    now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
    mode_prompt = MODES.get(mode_key, MODES["general"])["prompt"]
    parts = [BASE_SYSTEM_INSTRUCTION, f"Current date and time: {now}."]
    if mode_prompt:
        parts.append(mode_prompt)
    return "\n\n".join(parts)


# ============================================================
# 4. PERSISTENT STATE
# ============================================================

_state = {"chats": {}, "total_requests": 0}
_file_lock = threading.Lock()
chat_locks = defaultdict(asyncio.Lock)
rate_hits = defaultdict(deque)


def load_state() -> None:
    global _state
    try:
        if DATA_FILE.exists():
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _state = {
                    "chats": data.get("chats", {}),
                    "total_requests": data.get("total_requests", 0),
                }
                log.info("Loaded %d chats from %s", len(_state["chats"]), DATA_FILE)
    except Exception:
        log.exception("Could not load state, starting fresh")


def _write_file(payload: str) -> None:
    with _file_lock:
        tmp = DATA_FILE.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, DATA_FILE)


async def save_state() -> None:
    try:
        payload = json.dumps(_state, ensure_ascii=False)
        await asyncio.to_thread(_write_file, payload)
    except Exception:
        log.exception("Could not save state")


def get_chat(chat_id: int) -> dict:
    key = str(chat_id)
    chat = _state["chats"].get(key)
    if chat is None:
        chat = {}
        _state["chats"][key] = chat
    chat.setdefault("history", [])
    chat.setdefault("mode", "general")
    chat.setdefault("search", True)
    chat.setdefault("stopped", False)
    chat.setdefault("requests", 0)
    if chat["mode"] not in MODES:
        chat["mode"] = "general"
    return chat


# ============================================================
# 5. KEYBOARDS
# ============================================================

def main_menu(chat: dict) -> InlineKeyboardMarkup:
    mode = MODES[chat["mode"]]["label"]
    search_label = "🌐 Search: ON" if chat["search"] else "🌐 Search: OFF"
    if chat["stopped"]:
        power = InlineKeyboardButton("▶️ Resume", callback_data="resume")
    else:
        power = InlineKeyboardButton("🛑 Stop", callback_data="stop")

    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"🎭 Mode: {mode}", callback_data="modes")],
            [
                InlineKeyboardButton(search_label, callback_data="search"),
                power,
            ],
            [
                InlineKeyboardButton("🧹 New Chat", callback_data="clear"),
                InlineKeyboardButton("🆘 Help", callback_data="help"),
            ],
            [InlineKeyboardButton("🆔 My ID", callback_data="id")],
        ]
    )


def modes_menu(chat: dict) -> InlineKeyboardMarkup:
    rows = []
    row = []
    for key, mode in MODES.items():
        label = mode["label"] + (" ✅" if key == chat["mode"] else "")
        row.append(InlineKeyboardButton(label, callback_data=f"mode:{key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


# ============================================================
# 6. TEXT FORMATTING (Gemini Markdown -> Telegram HTML)
# ============================================================

def md_to_html(text: str) -> str:
    """Convert common Markdown to Telegram-safe HTML."""
    store = []

    def keep(fragment: str) -> str:
        store.append(fragment)
        return f"\x00{len(store) - 1}\x00"

    # fenced code blocks
    def code_block(m):
        lang = (m.group(1) or "").strip()
        code = html.escape(m.group(2).strip("\n"))
        if lang and re.fullmatch(r"[\w+#.\-]+", lang):
            return keep(f'<pre><code class="language-{lang}">{code}</code></pre>')
        return keep(f"<pre>{code}</pre>")

    text = re.sub(r"```([^\n`]*)\n?(.*?)```", code_block, text, flags=re.S)

    # markdown tables -> monospaced block
    def table(m):
        return keep("<pre>" + html.escape(m.group(1).rstrip()) + "</pre>")

    text = re.sub(
        r"(^[ \t]*\|.*\|[ \t]*(?:\n[ \t]*\|.*\|[ \t]*)+)", table, text, flags=re.M
    )

    # inline code
    text = re.sub(
        r"`([^`\n]+)`", lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text
    )

    # escape everything else
    text = html.escape(text, quote=False)

    # horizontal rules
    text = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "──────────", text, flags=re.M)

    # headers -> bold
    text = re.sub(
        r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$",
        lambda m: "<b>" + m.group(1).replace("**", "") + "</b>",
        text,
        flags=re.M,
    )

    # bullets
    text = re.sub(r"^([ \t]*)[*\-+][ \t]+", r"\1• ", text, flags=re.M)

    # bold / italic / strike
    text = re.sub(r"\*\*([^\n]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])__([^\n]+?)__(?![\w*])", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"~~([^\n]+?)~~", r"<s>\1</s>", text)

    # links
    text = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', text
    )

    # restore protected fragments
    text = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)
    return text.strip()


def split_markdown(text: str, limit: int = 3300) -> list:
    """Split long markdown into chunks without breaking code blocks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    lines = []
    for line in text.split("\n"):
        while len(line) > limit:
            lines.append(line[:limit])
            line = line[limit:]
        lines.append(line)

    chunks, cur, cur_len = [], [], 0
    in_code, lang = False, ""

    for line in lines:
        add = len(line) + 1
        if cur and cur_len + add > limit:
            if in_code:
                cur.append("```")
            chunks.append("\n".join(cur))
            cur = ["```" + lang] if in_code else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(line)
        cur_len += add
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code = False
            else:
                in_code = True
                lang = stripped[3:].strip()

    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if c.strip()]


# ============================================================
# 6b. TELEGRAM SAFE SENDERS
# ============================================================

def _retry_seconds(error: RetryAfter) -> float:
    value = error.retry_after
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


async def try_edit(message, text, parse_mode=None, markup=None, retries=2) -> bool:
    try:
        await message.edit_text(text, parse_mode=parse_mode, reply_markup=markup)
        return True
    except RetryAfter as e:
        if retries > 0:
            await asyncio.sleep(_retry_seconds(e) + 1)
            return await try_edit(message, text, parse_mode, markup, retries - 1)
        return False
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return True
        log.warning("Edit failed: %s", e)
        return False
    except TelegramError as e:
        log.warning("Edit failed: %s", e)
        return False


async def try_reply(message, text, parse_mode=None, markup=None, retries=2) -> bool:
    try:
        await message.reply_text(text, parse_mode=parse_mode, reply_markup=markup)
        return True
    except RetryAfter as e:
        if retries > 0:
            await asyncio.sleep(_retry_seconds(e) + 1)
            return await try_reply(message, text, parse_mode, markup, retries - 1)
        return False
    except TelegramError as e:
        log.warning("Reply failed: %s", e)
        return False


async def deliver_answer(status_msg, source_msg, text: str) -> None:
    """Replace the 'thinking' message with the final formatted answer."""
    chunks = split_markdown(text)
    if not chunks:
        chunks = ["❌ Empty response. Please try again."]

    for i, chunk in enumerate(chunks):
        html_text = md_to_html(chunk)
        ok = False
        if i == 0:
            ok = await try_edit(status_msg, html_text, ParseMode.HTML) or await try_edit(
                status_msg, chunk
            )
            if ok:
                continue
        ok = await try_reply(source_msg, html_text, ParseMode.HTML) or await try_reply(
            source_msg, chunk
        )
        if i == 0 and ok:
            try:
                await status_msg.delete()
            except TelegramError:
                pass


class LiveEdit:
    """Updates the 'thinking' message while Gemini streams the answer."""

    def __init__(self, message):
        self.message = message
        self.last_time = 0.0
        self.last_text = ""

    async def update(self, text: str) -> None:
        now = time.monotonic()
        if now < self.last_time + STREAM_INTERVAL:
            return
        shown = text[-3800:] + " ▌"
        if shown == self.last_text:
            return
        self.last_time = now
        try:
            await self.message.edit_text(shown)
            self.last_text = shown
        except RetryAfter as e:
            self.last_time = now + _retry_seconds(e)
        except TelegramError:
            pass


async def keep_typing(bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


# ============================================================
# 7. GEMINI ENGINE
# ============================================================

class GeminiError(Exception):
    """Error with a message that is safe to show to the user."""


class EmptyResponse(Exception):
    pass


TEMP_ERROR_WORDS = (
    "429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "DEADLINE", "TIMEOUT", "TIMED OUT", "OVERLOADED", "CONNECTION",
)


def build_contents(history: list, parts: list) -> list:
    messages = history[-MAX_HISTORY_MESSAGES:]
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]

    contents = [
        types.Content(
            role=m["role"],
            parts=[types.Part.from_text(text=m["text"])],
        )
        for m in messages
    ]
    contents.append(types.Content(role="user", parts=parts))
    return contents


def extract_sources(grounding_chunks) -> list:
    sources, seen = [], set()
    for item in grounding_chunks or []:
        web = getattr(item, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if not uri or uri in seen:
            continue
        seen.add(uri)
        title = re.sub(r"[\[\]()]", "", getattr(web, "title", "") or "Source")[:60]
        sources.append(f"[{title}]({uri})")
        if len(sources) >= 3:
            break
    return sources


async def stream_once(model, contents, config, on_update):
    text = ""
    grounding = None
    truncated = False

    stream = await client.aio.models.generate_content_stream(
        model=model, contents=contents, config=config
    )
    async for chunk in stream:
        try:
            piece = chunk.text
        except Exception:
            piece = None
        if piece:
            text += piece
            if on_update:
                await on_update(text)
        try:
            if chunk.candidates:
                cand = chunk.candidates[0]
                meta = getattr(cand, "grounding_metadata", None)
                if meta and getattr(meta, "grounding_chunks", None):
                    grounding = meta.grounding_chunks
                if "MAX_TOKENS" in str(getattr(cand, "finish_reason", "")):
                    truncated = True
        except Exception:
            pass

    if not text.strip():
        raise EmptyResponse()

    if truncated:
        text += "\n\n_(Answer was cut because it was too long. Send **continue** to get the rest.)_"
    sources = extract_sources(grounding)
    if sources:
        text += "\n\n🔎 **Sources:** " + " • ".join(sources)
    return text


async def generate(chat: dict, parts: list, on_update=None) -> str:
    contents = build_contents(chat["history"], parts)
    system_instruction = build_system_instruction(chat["mode"])
    use_search = ENABLE_SEARCH and chat["search"]
    last_error = ""

    for model in MODELS:
        attempt = 0
        while attempt < MAX_RETRIES:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=TEMPERATURE,
                top_p=0.95,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None,
            )
            try:
                return await stream_once(model, contents, config, on_update)

            except EmptyResponse:
                last_error = "empty response (possibly blocked by safety filters)"
                log.warning("[%s] empty response", model)
                break  # try next model

            except Exception as error:  # noqa: BLE001
                err = str(error)
                upper = err.upper()
                last_error = err
                log.warning("[%s] Gemini error (attempt %d): %s", model, attempt + 1, err[:300])

                # Search tool not supported by this model/key -> retry without it
                if use_search and any(w in upper for w in ("TOOL", "GROUNDING", "GOOGLE_SEARCH")) \
                        and "429" not in upper and "RESOURCE_EXHAUSTED" not in upper:
                    use_search = False
                    continue

                # Model name wrong / not available -> next model
                if "404" in upper or "NOT_FOUND" in upper or "NOT FOUND" in upper:
                    break

                if any(w in upper for w in TEMP_ERROR_WORDS):
                    attempt += 1
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    break  # next model

                break  # permanent error -> next model

    log.error("All models failed. Last error: %s", last_error[:500])
    if "API_KEY" in last_error.upper() or "PERMISSION_DENIED" in last_error.upper() or "401" in last_error:
        raise GeminiError("❌ The Gemini API key is invalid or has no permission. Please check the key.")
    if "SAFETY" in last_error.upper() or "empty response" in last_error:
        raise GeminiError("⚠️ I couldn't answer that one. Try rephrasing your message.")
    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
        raise GeminiError("⏳ The AI is very busy right now (quota limit). Please try again in a minute.")
    raise GeminiError("❌ Sorry, I couldn't process your request right now. Please try again in a few seconds.")


# ============================================================
# 8. USER INPUT (TEXT / PHOTO / FILE / VOICE)
# ============================================================

class UserInputError(Exception):
    pass


async def download_bytes(obj) -> bytes:
    size = getattr(obj, "file_size", None)
    if size and size > MAX_FILE_MB * 1024 * 1024:
        raise UserInputError(f"📦 File is too large. Maximum size is {MAX_FILE_MB} MB.")
    tg_file = await obj.get_file()
    data = await tg_file.download_as_bytearray()
    return bytes(data)


async def build_user_parts(msg):
    """Returns (gemini_parts, history_text)."""
    text = (msg.text or msg.caption or "").strip()
    parts = []
    label = ""

    if msg.photo:
        data = await download_bytes(msg.photo[-1])
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        label = "[Image] "
        text = text or (
            "Look at this image carefully. Describe what you see. If it contains a question, "
            "problem, code, or error message, solve it step by step."
        )

    elif msg.voice or msg.audio:
        media = msg.voice or msg.audio
        data = await download_bytes(media)
        mime = getattr(media, "mime_type", None) or "audio/ogg"
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        label = "[Voice] "
        text = text or (
            "This is a voice message from the user. Understand what they said and reply to it "
            "in the same language they spoke."
        )

    elif msg.document:
        doc = msg.document
        mime = (doc.mime_type or "").lower()
        name = doc.file_name or "file"
        data = await download_bytes(doc)

        if mime == "application/pdf" or mime.startswith(("image/", "audio/", "video/")):
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        else:
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                raise UserInputError(
                    "📄 I can read PDFs, images, audio and text/code files. This file type isn't supported."
                )
            if len(content) > 120_000:
                content = content[:120_000] + "\n...[file truncated]"
            parts.append(types.Part.from_text(text=f"File name: {name}\n\n{content}"))

        label = f"[File: {name}] "
        text = text or "Read this file carefully. Summarize it and point out anything important or wrong."

    if not text and not parts:
        return None, ""

    parts.append(types.Part.from_text(text=text))
    return parts, (label + text)[:4000]


# ============================================================
# 9. COMMANDS
# ============================================================

def start_text(name: str) -> str:
    return (
        f"👋 <b>Hi {html.escape(name)}, welcome to SolveMate AI!</b>\n\n"
        "🤖 I'm your smart AI assistant. I can help with:\n\n"
        "💻 Coding &amp; debugging\n"
        "📚 Study &amp; exams\n"
        "🧮 Math &amp; logic\n"
        "💼 Career, resume &amp; interviews\n"
        "✍️ Writing &amp; emails\n"
        "🛠️ Projects &amp; technology\n\n"
        "📸 Send a <b>photo</b>, 📄 <b>PDF/file</b> or 🎤 <b>voice message</b> - I understand them too.\n"
        "🌐 I can search the web for latest information.\n\n"
        "👇 <i>Just type your question, or choose an option.</i>"
    )


def help_text() -> str:
    return (
        "🆘 <b>SolveMate AI Help</b>\n\n"
        "💻 <b>Coding</b> - <code>Why is my Python code giving IndexError?</code>\n"
        "📚 <b>Study</b> - <code>Explain machine learning simply</code>\n"
        "🧮 <b>Math</b> - <code>Solve 25 x 18 step by step</code>\n"
        "💼 <b>Career</b> - <code>Prepare me for a Java interview</code>\n"
        "📸 <b>Image</b> - send a photo of a question/error\n"
        "📄 <b>File</b> - send a PDF or code file to analyse\n"
        "🎤 <b>Voice</b> - just speak your question\n\n"
        "<b>Commands</b>\n"
        "/start - Start the bot\n"
        "/menu - Open menu\n"
        "/mode - Choose expert mode\n"
        "/search - Turn web search on/off\n"
        "/clear - New chat (clear memory)\n"
        "/stop - Pause AI replies\n"
        "/resume - Resume AI replies\n"
        "/id - Show your Telegram ID\n\n"
        "💡 <i>Tip: In groups, mention me or reply to my message.</i>"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    chat["stopped"] = False
    name = update.effective_user.first_name if update.effective_user else "friend"
    await update.message.reply_text(
        start_text(name), reply_markup=main_menu(chat), parse_mode=ParseMode.HTML
    )
    await save_state()


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    await update.message.reply_text(
        "🏠 <b>SolveMate AI Menu</b>", reply_markup=main_menu(chat), parse_mode=ParseMode.HTML
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    await update.message.reply_text(
        help_text(), reply_markup=main_menu(chat), parse_mode=ParseMode.HTML
    )


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    await update.message.reply_text(
        "🎭 <b>Choose an expert mode:</b>", reply_markup=modes_menu(chat), parse_mode=ParseMode.HTML
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    chat["stopped"] = True
    await update.message.reply_text(
        "🛑 <b>SolveMate AI paused.</b>\n\nUse /resume to continue.",
        reply_markup=main_menu(chat),
        parse_mode=ParseMode.HTML,
    )
    await save_state()


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    chat["stopped"] = False
    await update.message.reply_text(
        "▶️ <b>SolveMate AI resumed!</b>\n\n🤖 Send me your question.",
        reply_markup=main_menu(chat),
        parse_mode=ParseMode.HTML,
    )
    await save_state()


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    chat["history"] = []
    await update.message.reply_text(
        "🧹 <b>New chat started.</b> Memory cleared.",
        reply_markup=main_menu(chat),
        parse_mode=ParseMode.HTML,
    )
    await save_state()


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    chat["search"] = not chat["search"]
    state = "ON ✅" if chat["search"] else "OFF ❌"
    await update.message.reply_text(
        f"🌐 Web search is now <b>{state}</b>",
        reply_markup=main_menu(chat),
        parse_mode=ParseMode.HTML,
    )
    await save_state()


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = get_chat(update.effective_chat.id)
    await update.message.reply_text(
        "🆔 <b>Your Telegram Information</b>\n\n"
        f"👤 User ID: <code>{update.effective_user.id}</code>\n"
        f"💬 Chat ID: <code>{update.effective_chat.id}</code>",
        reply_markup=main_menu(chat),
        parse_mode=ParseMode.HTML,
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    await update.message.reply_text(
        "📊 <b>Bot Stats</b>\n\n"
        f"👥 Chats: <code>{len(_state['chats'])}</code>\n"
        f"💬 Total requests: <code>{_state['total_requests']}</code>\n"
        f"🧠 Models: <code>{html.escape(', '.join(MODELS))}</code>",
        parse_mode=ParseMode.HTML,
    )


# ============================================================
# 10. BUTTON HANDLER
# ============================================================

async def safe_edit_query(query, text: str, markup) -> None:
    try:
        await query.edit_message_text(text, reply_markup=markup, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            log.warning("Button edit failed: %s", e)
    except TelegramError as e:
        log.warning("Button edit failed: %s", e)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except TelegramError:
        pass

    if not query.message:
        return

    chat_id = query.message.chat.id
    chat = get_chat(chat_id)
    data = query.data or ""

    if data == "stop":
        chat["stopped"] = True
        await safe_edit_query(
            query,
            "🛑 <b>SolveMate AI paused.</b>\n\nPress <b>Resume</b> to continue.",
            main_menu(chat),
        )

    elif data == "resume":
        chat["stopped"] = False
        await safe_edit_query(
            query,
            "▶️ <b>SolveMate AI resumed!</b>\n\n🤖 Send me your question.",
            main_menu(chat),
        )

    elif data == "help":
        await safe_edit_query(query, help_text(), main_menu(chat))

    elif data == "clear":
        chat["history"] = []
        await safe_edit_query(
            query, "🧹 <b>New chat started.</b> Memory cleared.", main_menu(chat)
        )

    elif data == "search":
        chat["search"] = not chat["search"]
        state = "ON ✅" if chat["search"] else "OFF ❌"
        await safe_edit_query(query, f"🌐 Web search is now <b>{state}</b>", main_menu(chat))

    elif data == "id":
        await safe_edit_query(
            query,
            "🆔 <b>Your Telegram Information</b>\n\n"
            f"👤 User ID: <code>{query.from_user.id}</code>\n"
            f"💬 Chat ID: <code>{chat_id}</code>",
            main_menu(chat),
        )

    elif data == "modes":
        await safe_edit_query(query, "🎭 <b>Choose an expert mode:</b>", modes_menu(chat))

    elif data.startswith("mode:"):
        key = data.split(":", 1)[1]
        if key in MODES:
            chat["mode"] = key
            await safe_edit_query(
                query,
                f"✅ Mode changed to <b>{html.escape(MODES[key]['label'])}</b>",
                main_menu(chat),
            )

    elif data == "menu":
        await safe_edit_query(query, "🏠 <b>SolveMate AI Menu</b>", main_menu(chat))

    await save_state()


# ============================================================
# 11. MESSAGE HANDLER
# ============================================================

def should_respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    msg = update.message
    if chat.type == "private":
        return True
    username = (context.bot.username or "").lower()
    text = (msg.text or msg.caption or "").lower()
    if username and f"@{username}" in text:
        return True
    reply = msg.reply_to_message
    if reply and reply.from_user and reply.from_user.id == context.bot.id:
        return True
    return False


def is_rate_limited(user_id: int) -> bool:
    now = time.monotonic()
    hits = rate_hits[user_id]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= REQUESTS_PER_MINUTE:
        return True
    hits.append(now)
    return False


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not update.effective_user:
        return
    if not should_respond(update, context):
        return

    chat_id = update.effective_chat.id
    chat = get_chat(chat_id)

    if chat["stopped"]:
        await msg.reply_text(
            "🛑 <b>SolveMate AI is paused.</b>\n\nUse the button or /resume to continue.",
            reply_markup=main_menu(chat),
            parse_mode=ParseMode.HTML,
        )
        return

    if is_rate_limited(update.effective_user.id):
        await msg.reply_text("⏳ You're sending messages too fast. Please wait a few seconds.")
        return

    async with chat_locks[chat_id]:
        typing_task = asyncio.create_task(keep_typing(context.bot, chat_id))
        status = None
        try:
            try:
                parts, history_text = await build_user_parts(msg)
            except UserInputError as e:
                await msg.reply_text(str(e))
                return
            if not parts:
                return

            status = await msg.reply_text("🤖 Thinking...")
            live = LiveEdit(status)

            try:
                answer = await generate(chat, parts, live.update)
            except GeminiError as e:
                await try_edit(status, str(e))
                return

            await deliver_answer(status, msg, answer)

            # remember conversation (text only)
            chat["history"].append({"role": "user", "text": history_text})
            chat["history"].append({"role": "model", "text": answer[:8000]})
            if len(chat["history"]) > MAX_HISTORY_MESSAGES:
                chat["history"] = chat["history"][-MAX_HISTORY_MESSAGES:]
            chat["requests"] += 1
            _state["total_requests"] += 1
            await save_state()

        except Exception:  # noqa: BLE001
            log.exception("Message handler error")
            if status is not None:
                await try_edit(status, "❌ Something went wrong. Please try again.")
            else:
                await try_reply(msg, "❌ Something went wrong. Please try again.")
        finally:
            typing_task.cancel()


# ============================================================
# 12. ERROR HANDLER
# ============================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    error = context.error
    if isinstance(error, NetworkError):
        log.warning("Network problem: %s", error)
        return
    log.error("Telegram error: %s", error, exc_info=error)


# ============================================================
# 13. STARTUP
# ============================================================

async def set_commands(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Start SolveMate AI"),
            BotCommand("menu", "Open menu"),
            BotCommand("mode", "Choose expert mode"),
            BotCommand("search", "Web search on/off"),
            BotCommand("clear", "New chat (clear memory)"),
            BotCommand("stop", "Pause AI replies"),
            BotCommand("resume", "Resume AI replies"),
            BotCommand("help", "Show help"),
            BotCommand("id", "Show my Telegram ID"),
        ]
    )
    log.info("Telegram command menu configured.")

    # Show which Gemini models your API key can actually use
    try:
        available = []
        async for m in await client.aio.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if name.startswith("gemini") and (not actions or "generateContent" in actions):
                available.append(name)
        log.info("Available Gemini models: %s", ", ".join(sorted(available)[:40]))
        missing = [m for m in MODELS if m not in available]
        if missing and available:
            log.warning("These configured models are NOT available: %s", ", ".join(missing))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not list Gemini models: %s", e)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN is missing! Please set the TELEGRAM_BOT_TOKEN environment variable.")
        sys.exit(1)

    if not GEMINI_API_KEY or "YOUR_" in GEMINI_API_KEY:
        log.error("GEMINI_API_KEY is missing! Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    load_state()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .post_init(set_commands)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("id", id_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.ALL | filters.VOICE | filters.AUDIO)
            & ~filters.COMMAND,
            message_handler,
        )
    )
    app.add_error_handler(error_handler)

    log.info("🤖 SolveMate AI is running | models: %s", ", ".join(MODELS))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()