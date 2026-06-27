"""LLM-based speech-to-text transcript cleanup via OpenAI."""

from __future__ import annotations

from openai import AsyncOpenAI

from autobot_stt.models.session import ChatMessage, Comment, Session

_MODEL = "gpt-4o-mini"
_TIMEOUT_SECONDS = 30.0

_SYSTEM_PROMPT = (
    "You correct speech-to-text transcription errors. You receive the user's "
    "existing draft, chat history, comments, and a raw Whisper transcript.\n\n"
    "Output ONLY the corrected spoken text to append to the draft — not the "
    "draft itself, not explanations. Preserve technical terms and meaning from "
    "the context. Fix homophones, punctuation, and word-boundary errors. "
    "Respect continuity with the existing draft when the user was mid-thought. "
    "Return plain text with no markdown or JSON wrapping."
)


def _format_chat_history(messages: list[ChatMessage]) -> str:
    if not messages:
        return "(none)"
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _format_comments(comments: list[Comment]) -> str:
    if not comments:
        return "(none)"
    return "\n".join(f"{c.author}: {c.body}" for c in comments)


def _build_user_message(session: Session) -> str:
    draft = session.draft_text.strip() or "(empty)"
    return (
        "## Existing draft (for continuity; do not repeat in output)\n"
        f"{draft}\n\n"
        "## Chat history\n"
        f"{_format_chat_history(session.chat_history)}\n\n"
        "## Comments\n"
        f"{_format_comments(session.comments)}\n\n"
        "## Raw transcript (fix this)\n"
        f"{session.raw_transcript}"
    )


async def cleanup_transcript(session: Session, *, api_key: str) -> str:
    """Return LLM-corrected spoken text for ``session.raw_transcript``.

    Args:
        session: Session containing draft/context/raw transcript.
        api_key: OpenAI API key.

    Returns:
        Stripped cleaned text ready to append to the user's draft.

    Raises:
        openai.OpenAIError: Propagated from the OpenAI API call.
    """
    async with AsyncOpenAI(api_key=api_key, timeout=_TIMEOUT_SECONDS) as client:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(session)},
            ],
        )
    if not response.choices:
        return ""
    content = response.choices[0].message.content or ""
    return content.strip()
