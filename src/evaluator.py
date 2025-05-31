from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.format import EvaluatorFormat
from src.state import OverallState, SubquestionResult
from typing import Dict, Any


# Global variables (will be set in main.py)
EVALUATOR_MODEL = None


def initialize_evaluator(cfg):
    """Initialize evaluator with configuration"""
    global EVALUATOR_MODEL
    
    model = ChatOpenAI(
        api_key=cfg.llm.api_key,
        model=cfg.llm.completions.model,
        temperature=cfg.llm.completions.temperature
    )
    EVALUATOR_MODEL = model.with_structured_output(EvaluatorFormat)


def format_query_results(query_results) -> str:
    """Format query results for presentation to the LLM"""
    formatted_results = []
    
    for result in query_results:
        source_type = result.source.upper()
        formatted_results.append(f"\n--- {source_type} QUERY: {result.query} ---")
        
        for i, content in enumerate(result.results, 1):
            formatted_results.append(f"{i}. {content}")
        
        formatted_results.append("")  # Empty line for separation
    
    return "\n".join(formatted_results)


def evaluator_node(state: OverallState) -> Dict[str, Any]:
    """
    Evaluate the collected information and generate an answer for the current subquestion
    """
    current_subquestion = state.get("current_subquestion", "")
    query_results = state.get("current_query_results", [])
    main_question = state.get("main_question", "")
    
    if not current_subquestion:
        return {"subquestion_results": state.get("subquestion_results", [])}
    
    # Format the collected information
    formatted_results = format_query_results(query_results)
    
    system_prompt = """You are an expert information evaluator. Your task is to analyze collected information and provide a comprehensive answer to a subquestion.

Guidelines for evaluation:
1. Answer: Provide a clear, comprehensive answer based on the available information
2. Confidence (0.0-1.0): Rate how confident you are in your answer based on:
   - Quality and reliability of sources
   - Consistency of information across sources
   - Completeness of information available
   - Clarity and specificity of the evidence
3. Support (0.0-1.0): Rate how much this subquestion and its answer contribute to answering the main question:
   - Direct relevance to the main question
   - Importance of this information for the overall answer
   - How much this fills gaps in understanding

Be objective and honest in your assessment. If information is limited or contradictory, reflect that in your confidence score."""

    human_prompt = f"""Please evaluate the following information and provide an answer to the subquestion:

MAIN QUESTION: {main_question}

SUBQUESTION: {current_subquestion}

COLLECTED INFORMATION:
{formatted_results}

Based on this information, please provide:
1. Your reasoning about the information quality and relevance
2. A comprehensive answer to the subquestion
3. A confidence score (0.0-1.0) for your answer
4. A support score (0.0-1.0) for how much this helps answer the main question"""

    try:
        response = EVALUATOR_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Create subquestion result
        subquestion_result = SubquestionResult(
            subquestion=current_subquestion,
            answer=response.answer,
            confidence=response.confidence,
            support=response.support,
            thoughts=response.thoughts
        )
        
        # Update subquestion results
        existing_results = state.get("subquestion_results", [])
        updated_results = existing_results + [subquestion_result]
        
        # Remove the processed subquestion from current_subquestions
        remaining_subquestions = state.get("current_subquestions", [])[1:]

        print("\n[EVALUATOR] Evaluating subquestion...")
        print(f"Evaluated subquestion: {current_subquestion}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}, Support: {response.support}")
        
        return {
            "subquestion_results": updated_results,
            "current_subquestions": remaining_subquestions
        }
        
    except Exception as e:
        print(f"Error in evaluator_node: {str(e)}")
        
        # Fallback: create a basic result
        fallback_result = SubquestionResult(
            subquestion=current_subquestion,
            answer=f"Unable to properly evaluate information due to error: {str(e)}",
            confidence=0.1,
            support=0.1,
            thoughts=f"Error occurred during evaluation: {str(e)}"
        )
        
        existing_results = state.get("subquestion_results", [])
        updated_results = existing_results + [fallback_result]
        remaining_subquestions = state.get("current_subquestions", [])[1:]
        
        return {
            "subquestion_results": updated_results,
            "current_subquestions": remaining_subquestions
        }