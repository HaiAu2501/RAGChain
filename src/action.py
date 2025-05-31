from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.format import ActionFormat
from src.state import OverallState
from typing import Dict, Any
from datetime import datetime


# Global variables (will be set in main.py)
ACTION_MODEL = None
N_QUERIES = None


def initialize_action(cfg):
    """Initialize action node with configuration"""
    global ACTION_MODEL, N_QUERIES
    
    model = ChatOpenAI(
        api_key=cfg.llm.api_key,
        model=cfg.llm.completions.model,
        temperature=cfg.llm.completions.temperature
    )
    ACTION_MODEL = model.with_structured_output(ActionFormat)
    N_QUERIES = cfg.tree.hyperparams.n_queries


def action_node(state: OverallState) -> Dict[str, Any]:
    """
    Generate database and web search queries for each subquestion
    This node processes one subquestion at a time and updates the state
    """
    current_subquestions = state.get("current_subquestions", [])
    
    if not current_subquestions:
        return {
            "current_thoughts": "",
            "current_database_queries": [],
            "current_web_queries": [],
            "current_subquestion": ""
        }
    
    # Get the first unprocessed subquestion
    subquestion = current_subquestions[0]
    
    # Get current time for context
    current_time = datetime.now().strftime("%B %Y")  # e.g., "December 2024"
    current_date = datetime.now().strftime("%Y-%m-%d")  # e.g., "2024-12-01"
    
    system_prompt = f"""You are an expert query generator. Your task is to create effective search queries to find information that answers a given subquestion.

IMPORTANT DATABASE INFORMATION:
- The database is a VECTOR DATABASE (semantic search), NOT a SQL database
- Database queries should be NATURAL LANGUAGE queries, not SQL commands
- Think of database queries as keywords or phrases that would find relevant documents
- Examples of GOOD database queries: "artificial intelligence recent developments", "machine learning applications 2024", "climate change impact agriculture"
- Examples of BAD database queries: "SELECT * FROM...", "WHERE year = 2024", any SQL syntax

CURRENT TIME CONTEXT:
- Current date: {current_date}
- Current month/year: {current_time}
- Use this information to create time-relevant queries when needed

Guidelines:
1. Generate up to {N_QUERIES} DATABASE QUERIES as natural language search terms for vector database
2. Generate up to {N_QUERIES} WEB SEARCH QUERIES optimized for web search engines
3. Database queries should focus on finding documents with relevant concepts, keywords, or topics
4. Web search queries should be optimized for finding recent information, current events, or latest data
5. Include relevant time context in queries when the question involves recent or current information
6. Make queries specific enough to find relevant information but not so narrow that they miss important content
7. Consider different phrasings or approaches for better coverage

Main question context: {state.get('main_question', 'Not provided')}"""

    human_prompt = f"""Generate effective search queries for this subquestion:

Subquestion: {subquestion}

Remember:
- Database queries: Natural language terms for semantic search (NOT SQL)
- Web queries: Optimized for search engines
- Include time context if the question involves recent developments or current information

Please provide your reasoning about what information is needed and then generate the appropriate database and web search queries."""

    try:
        response = ACTION_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Limit queries to N_QUERIES
        database_queries = response.database_queries[:N_QUERIES]
        web_queries = response.web_search_queries[:N_QUERIES]

        print("\n[ACTION] Generating queries for subquestion...")

        print(f"Action node generated {len(database_queries)} database queries and {len(web_queries)} web queries for subquestion: {subquestion}")
        print("Database Queries:", database_queries)
        print("Web Queries:", web_queries)

        return {
            "current_thoughts": response.thoughts,
            "current_database_queries": database_queries,
            "current_web_queries": web_queries,
            "current_subquestion": subquestion
        }
        
    except Exception as e:
        print(f"Error in action_node: {str(e)}")
        # Fallback: create simple queries based on the subquestion
        return {
            "current_thoughts": f"Error occurred, using fallback queries for: {subquestion}",
            "current_database_queries": [subquestion],
            "current_web_queries": [subquestion],
            "current_subquestion": subquestion
        }