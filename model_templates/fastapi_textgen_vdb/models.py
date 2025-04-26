from pydantic import BaseModel, Field
from typing import List, Optional
# Pydantic models for tools/call endpoint
class ToolCallArguments(BaseModel):
    query: str

class ToolCallParams(BaseModel):
    name: str
    arguments: ToolCallArguments

class ToolCallRequest(BaseModel):
    id: int
    method: str = "tools/call"
    params: ToolCallParams

class TextContent(BaseModel):
    type: str = "text"
    text: str

class ToolCallResult(BaseModel):
    content: List[TextContent]
    isError: bool = False

class ToolCallResponse(BaseModel):
    id: int
    result: ToolCallResult
