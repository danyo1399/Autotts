from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class Comment(BaseModel):
    author: str
    body: str


class CreateSessionRequest(BaseModel):
    draft_text: str = ""
    chat_history: list[ChatMessage] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    session_id: str


class Session(BaseModel):
    id: str
    draft_text: str
    chat_history: list[ChatMessage]
    comments: list[Comment]
    created_at: datetime
    raw_transcript: str = ""
    partial_transcripts: list[str] = Field(default_factory=list)
