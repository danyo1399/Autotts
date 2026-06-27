"""WebSocket streaming endpoint for live speech-to-text transcription."""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from autobot_stt.config import Settings, get_settings
from autobot_stt.dependencies.auth import check_ws_api_key
from autobot_stt.dependencies.store import get_session_store
from autobot_stt.models.session import Session
from autobot_stt.services.audio_decoder import AudioDecodeError, decode_webm_opus_to_pcm
from autobot_stt.services.whisper_service import WhisperService, build_initial_prompt
from autobot_stt.stores.base import SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

STREAM_CHUNK_SECONDS = 2.0
STREAM_SILENCE_TIMEOUT_SECONDS = 1.5
PCM_SAMPLE_RATE = 16000
STREAM_FLUSH_SAMPLES = int(STREAM_CHUNK_SECONDS * PCM_SAMPLE_RATE)
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

    whisper = _get_whisper_service(websocket)
    whisper_lock = _get_whisper_lock(websocket)
    initial_prompt = _build_initial_prompt(session)

    webm_buffer: bytearray = bytearray()

    async def flush(force: bool) -> None:
        """Decode buffered WebM; transcribe and emit when threshold met or ``force``."""
        if not webm_buffer:
            return
        if not force and len(webm_buffer) < STREAM_MIN_BYTES_FOR_DECODE:
            return

        batch = bytes(webm_buffer)
        try:
            pcm = await asyncio.to_thread(decode_webm_opus_to_pcm, batch)
        except AudioDecodeError as exc:
            logger.warning("audio decode failed: %s", exc)
            webm_buffer.clear()
            await websocket.send_json(
                {"type": "error", "message": "Failed to decode audio"}
            )
            return
        except FileNotFoundError:
            logger.error("ffmpeg unavailable; cannot decode audio")
            webm_buffer.clear()
            await websocket.send_json(
                {"type": "error", "message": "Audio decoder unavailable"}
            )
            return

        should_transcribe = force or len(pcm) >= STREAM_FLUSH_SAMPLES
        if not should_transcribe:
            return

        webm_buffer.clear()
        await _transcribe_and_emit(
            pcm, websocket, whisper, whisper_lock, session, initial_prompt
        )

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=STREAM_SILENCE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                await flush(force=True)
                continue

            if message["type"] == "websocket.disconnect":
                break

            bytes_data = message.get("bytes")
            if not bytes_data:
                continue

            webm_buffer.extend(bytes_data)
            await flush(force=False)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        logger.info("websocket runtime error: %s", exc)
    finally:
        if webm_buffer:
            try:
                await flush(force=True)
            except (RuntimeError, WebSocketDisconnect):
                logger.debug("trailing flush skipped; client disconnected")


async def _transcribe_and_emit(
    pcm: np.ndarray,
    websocket: WebSocket,
    whisper: WhisperService,
    whisper_lock: asyncio.Lock,
    session: Session,
    initial_prompt: str | None,
) -> None:
    if len(pcm) == 0:
        return

    async with whisper_lock:
        try:
            text = await asyncio.to_thread(whisper.transcribe, pcm, initial_prompt)
        except Exception as exc:  # noqa: BLE001 - log and recover, do not crash stream
            logger.exception("whisper transcribe failed: %s", exc)
            await websocket.send_json(
                {"type": "error", "message": "Transcription failed"}
            )
            return

    if not text:
        return

    if session.raw_transcript:
        session.raw_transcript += " "
    session.raw_transcript += text
    session.partial_transcripts.append(text)

    await websocket.send_json(
        {
            "type": "partial_transcript",
            "text": session.raw_transcript,
            "is_final": False,
        }
    )


def _get_whisper_service(websocket: WebSocket) -> WhisperService:
    service: WhisperService | None = getattr(websocket.app.state, "whisper_service", None)
    if service is None:
        raise RuntimeError("Whisper service is not initialized on app.state")
    return service


def _get_whisper_lock(websocket: WebSocket) -> asyncio.Lock:
    lock: asyncio.Lock | None = getattr(websocket.app.state, "whisper_lock", None)
    if lock is None:
        raise RuntimeError("Whisper lock is not initialized on app.state")
    return lock


def _build_initial_prompt(session: Session) -> str | None:
    history: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in session.chat_history
    ]
    return build_initial_prompt(session.draft_text, history) or None
