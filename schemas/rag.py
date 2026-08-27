from pydantic import BaseModel, Field
from typing import Any, Optional


class QuestionRequest(BaseModel):
    question: str
    metadata: Optional[dict[str, Any]] = None
    upload_id: Optional[str] = None
    session_id: Optional[str] = None
    user_name: Optional[str] = None
    user_id: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: Optional[str] = None
    upload_id: Optional[str] = None
    user_name: Optional[str] = None
    user_id: Optional[str] = None
    available_users: list[dict[str, str]] = Field(default_factory=list)
