import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response

from autobot_stt.dependencies.store import get_session_store
from autobot_stt.models.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    Session,
)
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
