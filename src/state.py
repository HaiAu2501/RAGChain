from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SubquestionResult:
    """Container for subquestion and its evaluation result"""
    subquestion: str
    answer: str
    confidence: float
    support: float
    thoughts: str


@dataclass
class QueryResult:
    """Container for query results from database or web search"""
    query: str
    results: List[str]
    source: str  # "database" or "web"


class InputState(TypedDict):
    """Input state for the workflow"""
    question: str


class OutputState(TypedDict):
    """Output state for the workflow"""
    final_answer: str


class OverallState(InputState, OutputState):
    """Complete state maintained throughout the workflow"""
    # Main question tracking
    main_question: str
    
    # Current iteration tracking
    current_iteration: int
    max_iterations: int
    
    # Current subquestions to process
    current_subquestions: List[str]
    
    # Results from each subquestion evaluation
    subquestion_results: List[SubquestionResult]
    
    # Query results for current processing
    current_query_results: List[QueryResult]
    
    # Current subquestion being processed
    current_subquestion: str
    
    # Action results for current subquestion
    current_thoughts: str
    current_database_queries: List[str]
    current_web_queries: List[str]
    
    # Final answer when ready
    final_answer: Optional[str]
    
    # Flag to indicate completion
    is_complete: bool