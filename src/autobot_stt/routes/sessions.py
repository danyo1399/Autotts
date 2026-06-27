import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response

from autobot_stt.config import Settings, get_settings
from autobot_stt.dependencies.store import get_session_store
from autobot_stt.models.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    FinalizeSessionResponse,
    Session,
)
from autobot_stt.services.llm_cleanup import cleanup_transcript
from autobot_stt.stores.base import SessionStore

router = APIRouter(tags=["sessions"])


@router.post("/sessions", status_code=201, response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    store: SessionStore = Depends(get_session_store),
) -> CreateSessionResponse:
    session = Session(
        id=str(uuid.uuid4()),
        draft_text=body.draft_text,
        chat_history=body.chat_history,
        comments=body.comments,
        created_at=datetime.now(UTC),
    )
    await store.create(session)
    return CreateSessionResponse(session_id=session.id)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> Response:
    deleted = await store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


@router.post(
    "/sessions/{session_id}/finalize",
    response_model=FinalizeSessionResponse,
)
async def finalize_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> FinalizeSessionResponse:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.raw_transcript.strip():
        raise HTTPException(status_code=400, detail="Session has no transcript")
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    raw_transcript = session.raw_transcript
    cleaned = await cleanup_transcript(session, api_key=settings.openai_api_key)
    await store.delete(session_id)
    return FinalizeSessionResponse(text=cleaned, raw_transcript=raw_transcript)
