"""WebSocket streaming endpoint for live speech-to-text transcription."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from autobot_stt.config import Settings, get_settings
from autobot_stt.dependencies.auth import check_ws_api_key
from autobot_stt.dependencies.store import (
    get_session_store,
    get_whisper_lock,
    get_whisper_service,
)
from autobot_stt.models.session import Session
from autobot_stt.services.audio_decoder import AudioDecodeError, decode_webm_opus_to_pcm
from autobot_stt.services.whisper_service import WhisperService, build_initial_prompt
from autobot_stt.stores.base import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])

STREAM_CHUNK_SECONDS = 2
STREAM_SILENCE_TIMEOUT_SECONDS = 1.5
PCM_SAMPLE_RATE = 16000
STREAM_FLUSH_SAMPLES = STREAM_CHUNK_SECONDS * PCM_SAMPLE_RATE
STREAM_MIN_BYTES_FOR_DECODE = 1024

WS_CLOSE_AUTH_FAILURE = 4401
WS_CLOSE_SESSION_NOT_FOUND = 4404


@router.websocket("/sessions/{session_id}/stream")
async def stream_session(
    websocket: WebSocket,
    session_id: str,
    token: str | None = None,
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
    whisper: WhisperService = Depends(get_whisper_service),
    whisper_lock: asyncio.Lock = Depends(get_whisper_lock),
) -> None:
    """Stream binary WebM/Opus chunks; emit partial transcripts and accumulate text."""
    if not check_ws_api_key(websocket, token, settings):
        await websocket.close(code=WS_CLOSE_AUTH_FAILURE, reason="Unauthorized")
        return

    session = await store.get(session_id)
    if session is None:
        await websocket.close(code=WS_CLOSE_SESSION_NOT_FOUND, reason="Session not found")
        return

    await websocket.accept()
    await websocket.send_json({"type": "ready", "session_id": session_id})

    initial_prompt = _build_initial_prompt(session)
    webm_buffer: bytearray = bytearray()

    async def flush(force: bool) -> bool:
        """Decode buffered WebM; transcribe and emit when threshold met or ``force``.

        Returns False when the session was removed (finalize/delete); callers
        should stop streaming.
        """
        if not webm_buffer:
            return True
        if not force and len(webm_buffer) < STREAM_MIN_BYTES_FOR_DECODE:
            return True

        batch = bytes(webm_buffer)
        try:
            pcm = await asyncio.to_thread(decode_webm_opus_to_pcm, batch)
        except AudioDecodeError as exc:
            if not force:
                # MediaRecorder fragments may not decode until more bytes arrive.
                logger.debug("audio decode deferred (incomplete batch): %s", exc)
                return True
            logger.warning("audio decode failed: %s", exc)
            webm_buffer.clear()
            await websocket.send_json({"type": "error", "message": "Failed to decode audio"})
            return True
        except FileNotFoundError:
            logger.error("ffmpeg unavailable; cannot decode audio")
            webm_buffer.clear()
            await websocket.send_json({"type": "error", "message": "Audio decoder unavailable"})
            return True

        if not force and len(pcm) < STREAM_FLUSH_SAMPLES:
            return True

        webm_buffer.clear()
        return await _transcribe_and_emit(
            pcm,
            websocket,
            whisper,
            whisper_lock,
            session,
            store,
            initial_prompt,
        )

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=STREAM_SILENCE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                if not await flush(force=True):
                    break
                continue

            if message["type"] == "websocket.disconnect":
                break

            bytes_data = message.get("bytes")
            if not bytes_data:
                continue

            webm_buffer.extend(bytes_data)
            if not await flush(force=False):
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        logger.info("websocket runtime error: %s", exc)
    finally:
        if webm_buffer:
            try:
                await flush(force=True)
            except Exception:  # noqa: BLE001 - cleanup path must not mask in-flight errors
                logger.debug("trailing flush skipped", exc_info=True)


async def _transcribe_and_emit(
    pcm: np.ndarray,
    websocket: WebSocket,
    whisper: WhisperService,
    whisper_lock: asyncio.Lock,
    session: Session,
    store: SessionStore,
    initial_prompt: str | None,
) -> bool:
    """Transcribe PCM and emit a partial transcript.

    Returns False when the session was removed from the store (finalize/delete).
    """
    if len(pcm) == 0:
        return True

    # Hold whisper_lock across transcribe AND the transcript read-modify-write.
    # The mutation has no `await` today so the GIL keeps it atomic, but a
    # future await between these statements would let two concurrent
    # connections to the same session interleave and lose text. Locking
    # makes the critical section unambiguously safe.
    transcribe_error = False
    cumulative: str | None = None
    async with whisper_lock:
        if await store.get(session.id) is None:
            return False

        try:
            text = await asyncio.to_thread(whisper.transcribe, pcm, initial_prompt)
        except Exception:  # noqa: BLE001 - log and recover, do not crash stream
            logger.exception("whisper transcribe failed")
            transcribe_error = True
        else:
            if text:
                if await store.get(session.id) is None:
                    return False
                if session.raw_transcript:
                    session.raw_transcript += " "
                session.raw_transcript += text
                session.partial_transcripts.append(text)
                cumulative = session.raw_transcript

    # Send outside the lock so a slow client cannot stall other transcriptions.
    if transcribe_error:
        await websocket.send_json({"type": "error", "message": "Transcription failed"})
        return True

    if cumulative is not None:
        await websocket.send_json(
            {
                "type": "partial_transcript",
                "text": cumulative,
                "is_final": False,
            }
        )
    return True


def _build_initial_prompt(session: Session) -> str | None:
    history: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in session.chat_history
    ]
    prompt = build_initial_prompt(session.draft_text, history)
    return prompt or None
