from pydantic import BaseModel, HttpUrl
from typing import List

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class FinalAnswer(BaseModel):
    answer: str

class QueryResponse(BaseModel):
    answers: List[str]