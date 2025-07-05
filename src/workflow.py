from langgraph.graph import StateGraph, END, START
from src.state import OverallState, InputState, OutputState
from src.decomposer import decomposer_node, initialize_decomposer
from src.action import action_node, initialize_action
from src.observation import observation_node, initialize_observation
from src.evaluator import evaluator_node, initialize_evaluator
from src.merger import merger_node, initialize_merger
from typing import Dict, Any


def should_continue_subquestions(state: OverallState) -> str:
    """Routing function to determine if we should continue processing subquestions"""
    current_subquestions = state.get("current_subquestions", [])
    
    if current_subquestions:
        # Still have subquestions to process
        return "action"
    else:
        # No more subquestions, go to merger
        return "merger"


def should_continue_workflow(state: OverallState) -> str:
    """Routing function to determine if workflow should continue or end"""
    is_complete = state.get("is_complete", False)
    
    if is_complete:
        return END
    else:
        return "decomposer"


def create_workflow(cfg, vectordb, bm25_index=None):
    """Create and configure the LangGraph workflow"""
    
    # Initialize all nodes with configuration
    initialize_decomposer(cfg)
    initialize_action(cfg)
    initialize_observation(cfg, vectordb, bm25_index)
    initialize_evaluator(cfg)
    initialize_merger(cfg)
    
    # Create the state graph
    workflow = StateGraph(OverallState, input=InputState, output=OutputState)
    
    # Add nodes to the workflow
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("action", action_node)
    workflow.add_node("observation", observation_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("merger", merger_node)
    
    # Define the workflow flow
    # Start with decomposer
    workflow.add_edge(START, "decomposer")
    
    # From decomposer, go to action (to process first subquestion)
    workflow.add_edge("decomposer", "action")
    
    # From action, go to observation
    workflow.add_edge("action", "observation")
    
    # From observation, go to evaluator
    workflow.add_edge("observation", "evaluator")
    
    # From evaluator, decide whether to continue with more subquestions or go to merger
    workflow.add_conditional_edges(
        "evaluator",
        should_continue_subquestions,
        {
            "action": "action",     # More subquestions to process
            "merger": "merger"      # All subquestions processed, go to merger
        }
    )
    
    # From merger, decide whether to end or continue with new iteration
    workflow.add_conditional_edges(
        "merger",
        should_continue_workflow,
        {
            "decomposer": "decomposer",  # Continue with new iteration
            END: END                     # Workflow complete
        }
    )
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow


def initialize_state(question: str, max_iterations: int) -> OverallState:
    """Initialize the workflow state with the input question"""
    return {
        "question": question,
        "main_question": question,
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "current_subquestions": [],
        "subquestion_results": [],
        "current_query_results": [],
        "current_subquestion": "",
        "current_thoughts": "",
        "current_database_queries": [],
        "current_web_queries": [],
        "final_answer": None,
        "is_complete": False
    }