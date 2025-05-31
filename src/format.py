from pydantic import BaseModel, Field
from typing import List, Optional


class DecomposerFormat(BaseModel):
    """Format for Decomposer Node output"""
    thoughts: str = Field(
        default="",
        description="Reasoning about how to decompose the main question"
    )
    subquestions: List[str] = Field(
        default_factory=list,
        description="List of subquestions derived from the main question"
    )


class ActionFormat(BaseModel):
    """Format for Action Node output"""
    thoughts: str = Field(
        default="",
        description="Reasoning about what queries are needed for this subquestion"
    )
    database_queries: List[str] = Field(
        default_factory=list,
        description="Natural language queries for vector database semantic search (NOT SQL queries)"
    )
    web_search_queries: List[str] = Field(
        default_factory=list,
        description="Search engine optimized queries for web search"
    )


class EvaluatorFormat(BaseModel):
    """Format for Evaluator Node output"""
    thoughts: str = Field(
        default="",
        description="Reasoning about the answer quality and relevance"
    )
    answer: str = Field(
        default="",
        description="Answer to the subquestion based on collected information"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score of the answer quality (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    support: float = Field(
        default=0.0,
        description="How much this subquestion supports answering the main question (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )


class MergerFormat(BaseModel):
    """Format for Merger Node output"""
    thoughts: str = Field(
        default="",
        description="Reasoning about whether enough information is available for final answer"
    )
    has_final_answer: bool = Field(
        default=False,
        description="Whether sufficient information is available to provide final answer"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer to the main question if available"
    )
    selected_subquestion: Optional[str] = Field(
        default=None,
        description="Selected subquestion for further decomposition if final answer not ready"
    )
    new_subquestions: List[str] = Field(
        default_factory=list,
        description="New subquestions generated from the selected subquestion"
    )