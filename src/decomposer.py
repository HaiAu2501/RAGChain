from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.format import DecomposerFormat
from src.state import OverallState
from typing import Dict, Any
from datetime import datetime
import os


# Global variables (will be set in main.py)
DECOMPOSER_MODEL = None
N_BRANCHES = None


def initialize_decomposer(cfg):
    """Initialize decomposer with configuration"""
    global DECOMPOSER_MODEL, N_BRANCHES
    
    model = ChatOpenAI(
        api_key=cfg.llm.api_key,
        model=cfg.llm.completions.model,
        temperature=cfg.llm.completions.temperature
    )
    DECOMPOSER_MODEL = model.with_structured_output(DecomposerFormat)
    N_BRANCHES = cfg.tree.hyperparams.n_branches


def decomposer_node(state: OverallState) -> Dict[str, Any]:
    """
    Decompose the main question or selected subquestion into subquestions
    """
    # Determine what question to decompose
    if state.get("current_iteration", 0) == 0:
        # First iteration - decompose main question
        question_to_decompose = state["question"]
        context = "This is the main question that needs to be answered."
    else:
        # Later iterations - decompose selected subquestion
        # Find the selected subquestion from merger results
        question_to_decompose = state.get("current_subquestion", state["question"])
        context = f"""This is a subquestion selected for further decomposition. 
        Main question: {state['main_question']}
        Previous subquestions and answers: {[f'Q: {r.subquestion}, A: {r.answer}' for r in state.get('subquestion_results', [])]}"""
    
    # Get current time for context
    current_time = datetime.now().strftime("%B %Y")  # e.g., "December 2024"
    current_date = datetime.now().strftime("%Y-%m-%d")  # e.g., "2024-12-01"
    
    system_prompt = f"""You are an expert question decomposer. Your task is to break down complex questions into simpler, more focused subquestions that can be answered independently.

CURRENT TIME CONTEXT:
- Current date: {current_date}
- Current month/year: {current_time}
- Use this information when creating time-relevant subquestions

Guidelines:
1. Generate exactly {N_BRANCHES} subquestions maximum
2. Each subquestion should be specific and answerable through database search or web search
3. Subquestions should collectively help answer the original question
4. Avoid redundant or overly similar subquestions
5. Focus on the most important aspects needed to answer the original question
6. Include time context in subquestions when the original question involves recent developments, current events, or time-sensitive information
7. For questions about "latest", "recent", "current" topics, ensure subquestions specify relevant time periods

Examples of time-aware subquestions:
- "What are the latest AI developments in 2024?"
- "What recent climate policies have been implemented?"
- "How has renewable energy adoption changed in recent years?"

Context: {context}"""

    human_prompt = f"""Please decompose this question into focused subquestions:
    
Question: {question_to_decompose}

Consider the current time context ({current_time}) when creating subquestions, especially if the question involves recent developments, current trends, or time-sensitive information.

Provide your reasoning and then list the subquestions."""

    try:
        response = DECOMPOSER_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Limit to N_BRANCHES subquestions
        subquestions = response.subquestions[:N_BRANCHES]

        print("\n[DECOMPOSER] Decomposing question...")

        print(f"Decomposed {len(subquestions)} subquestions from '{question_to_decompose}':")
        for i, subq in enumerate(subquestions):
            print(f"  {i + 1}. {subq}")
        
        return {
            "current_subquestions": subquestions,
            "main_question": state.get("main_question", state["question"]),
            "current_iteration": state.get("current_iteration", 0)
        }
        
    except Exception as e:
        print(f"Error in decomposer_node: {str(e)}")
        # Fallback: use original question as single subquestion
        return {
            "current_subquestions": [question_to_decompose],
            "main_question": state.get("main_question", state["question"]),
            "current_iteration": state.get("current_iteration", 0)
        }