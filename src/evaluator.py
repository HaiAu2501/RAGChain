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
    
    system_prompt = """You are an expert information evaluator with a critical and rigorous approach. Your task is to analyze collected information and provide a comprehensive answer to a subquestion.

CRITICAL EVALUATION GUIDELINES:
1. SOURCE VERIFICATION: Carefully verify if sources match the requirements in the main question
   - If the question specifies particular sources (e.g., "as reported by X and Y"), ensure information comes from those exact sources
   - Distinguish between different publications and their coverage
   - Flag when sources don't match the question's requirements

2. INFORMATION ACCURACY: Be extremely cautious about factual claims
   - Cross-reference information across multiple sources
   - Identify contradictions or inconsistencies
   - Verify specific details like names, dates, charges, and circumstances
   - Be skeptical of unverified claims

3. ANSWER COMPLETENESS: Ensure the answer directly addresses the subquestion
   - Answer must be specific and precise
   - Avoid generic or vague responses
   - Include relevant context and details

4. CONFIDENCE SCORING (0.0-1.0) - BE STRICT:
   - 0.9-1.0: Multiple high-quality sources with consistent, verified information
   - 0.7-0.8: Good sources with mostly consistent information, minor gaps
   - 0.5-0.6: Mixed quality sources or some inconsistencies
   - 0.3-0.4: Limited sources or significant inconsistencies
   - 0.0-0.2: Poor sources, major contradictions, or insufficient information

5. SUPPORT SCORING (0.0-1.0) - Rate relevance to main question:
   - Does this directly address the main question's requirements?
   - How essential is this information for the complete answer?
   - Does it fill critical gaps in understanding?

IMPORTANT: If the collected information doesn't meet the specific requirements of the main question (e.g., wrong sources, wrong person, wrong details), give LOW confidence scores and explain why in your reasoning."""

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