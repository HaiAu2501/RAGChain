from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.format import MergerFormat
from src.state import OverallState
from typing import Dict, Any


# Global variables (will be set in main.py)
MERGER_MODEL = None
N_BRANCHES = None
MAX_ITERATIONS = None


def initialize_merger(cfg):
    """Initialize merger with configuration"""
    global MERGER_MODEL, N_BRANCHES, MAX_ITERATIONS
    
    model = ChatOpenAI(
        api_key=cfg.llm.api_key,
        model=cfg.llm.completions.model,
        temperature=cfg.llm.completions.temperature
    )
    MERGER_MODEL = model.with_structured_output(MergerFormat)
    N_BRANCHES = cfg.tree.hyperparams.n_branches
    MAX_ITERATIONS = cfg.tree.hyperparams.n_iterations


def format_subquestion_results(results) -> str:
    """Format subquestion results for presentation to the LLM"""
    formatted = []
    
    for i, result in enumerate(results, 1):
        formatted.append(f"SUBQUESTION {i}: {result.subquestion}")
        formatted.append(f"ANSWER: {result.answer}")
        formatted.append(f"CONFIDENCE: {result.confidence:.2f}")
        formatted.append(f"SUPPORT: {result.support:.2f}")
        formatted.append(f"REASONING: {result.thoughts}")
        formatted.append("---")
    
    return "\n".join(formatted)


def merger_node(state: OverallState) -> Dict[str, Any]:
    """
    Decide whether to provide final answer or continue with more subquestions
    """
    main_question = state.get("main_question", "")
    subquestion_results = state.get("subquestion_results", [])
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    
    # Check if we've reached maximum iterations
    if current_iteration >= max_iterations:
        # Force final answer
        return force_final_answer(main_question, subquestion_results)
    
    # Format all collected information
    formatted_results = format_subquestion_results(subquestion_results)
    
    system_prompt = f"""You are an expert answer synthesizer. Your task is to determine whether enough information has been collected to answer the main question, or if more focused research is needed.

Guidelines:
1. If you have sufficient information to provide a comprehensive answer to the main question:
   - Set has_final_answer to True
   - Provide a complete final_answer that addresses all aspects of the main question
   - Synthesize information from all subquestions into a coherent response

2. If more information is needed:
   - Set has_final_answer to False
   - Select the subquestion with the HIGHEST combination of confidence and support scores
   - Generate up to {N_BRANCHES} new focused subquestions that dive deeper into that topic
   - New subquestions should be more specific and targeted than previous ones

Current iteration: {current_iteration + 1}/{max_iterations}"""

    human_prompt = f"""Based on the collected information, decide whether to provide a final answer or continue research:

MAIN QUESTION: {main_question}

COLLECTED INFORMATION FROM SUBQUESTIONS:
{formatted_results}

Please analyze whether this information is sufficient to comprehensively answer the main question. If not, select the most promising subquestion for further exploration and generate new focused subquestions."""

    try:
        response = MERGER_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        if response.has_final_answer:
            return {
                "final_answer": response.final_answer,
                "is_complete": True
            }
        else:
            # Continue with more iterations
            # Limit new subquestions to N_BRANCHES
            new_subquestions = response.new_subquestions[:N_BRANCHES]
            
            return {
                "current_subquestions": new_subquestions,
                "current_iteration": current_iteration + 1,
                "current_subquestion": response.selected_subquestion,
                "is_complete": False
            }
        
    except Exception as e:
        print(f"Error in merger_node: {str(e)}")
        # Fallback: provide final answer based on available information
        return force_final_answer(main_question, subquestion_results)


def force_final_answer(main_question: str, subquestion_results) -> Dict[str, Any]:
    """Create a final answer when forced to conclude"""
    if not subquestion_results:
        final_answer = "I don't have sufficient information to answer this question comprehensively."
    else:
        # Combine all answers
        answers = []
        for result in subquestion_results:
            answers.append(f"Regarding '{result.subquestion}': {result.answer}")
        
        final_answer = f"Based on the available information:\n\n" + "\n\n".join(answers)
        final_answer += f"\n\nThis represents the most comprehensive answer I can provide for: {main_question}"
    
    return {
        "final_answer": final_answer,
        "is_complete": True
    }