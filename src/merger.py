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
        formatted.append(f"CÂU HỎI PHỤ {i}: {result.subquestion}")
        formatted.append(f"TRẢ LỜI: {result.answer}")
        formatted.append(f"ĐỘ TIN CẬY: {result.confidence:.2f}")
        formatted.append(f"ĐỘ HỖ TRỢ: {result.support:.2f}")
        formatted.append(f"LÝ DO: {result.thoughts}")
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
    
    system_prompt = f"""Bạn là một chuyên gia tổng hợp câu trả lời. Nhiệm vụ của bạn là xác định xem đã thu thập đủ thông tin để trả lời câu hỏi chính hay cần nghiên cứu thêm chuyên sâu.

Hướng dẫn:
1. Nếu bạn có đủ thông tin để đưa ra câu trả lời toàn diện cho câu hỏi chính:
   - Đặt has_final_answer thành True
   - Cung cấp một final_answer hoàn chỉnh giải quyết tất cả các khía cạnh của câu hỏi chính
   - Tổng hợp thông tin từ tất cả các câu hỏi phụ thành một phản hồi mạch lạc

2. Nếu cần thêm thông tin:
   - Đặt has_final_answer thành False
   - Chọn câu hỏi phụ có điểm số KẾT HỢP CỦA TIN CẬY VÀ HỖ TRỢ cao nhất
   - Tạo ra tối đa {N_BRANCHES} câu hỏi phụ mới tập trung sâu hơn vào chủ đề đó
   - Các câu hỏi phụ mới phải cụ thể và nhắm mục tiêu hơn so với các câu trước

Lượt hiện tại: {current_iteration + 1}/{max_iterations}

Vui lòng trả lời bằng tiếng Việt."""

    human_prompt = f"""Dựa trên thông tin đã thu thập, hãy quyết định có nên đưa ra câu trả lời cuối cùng hay tiếp tục nghiên cứu:

CÂU HỎI CHÍNH: {main_question}

THÔNG TIN ĐÃ THU THẬP TỪ CÁC CÂU HỎI PHỤ:
{formatted_results}

Vui lòng phân tích xem thông tin này có đủ để trả lời toàn diện câu hỏi chính hay không. Nếu chưa, hãy chọn câu hỏi phụ hứa hẹn nhất để khám phá thêm và tạo ra các câu hỏi phụ mới tập trung hơn."""

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
        print(f"Lỗi trong merger_node: {str(e)}")
        # Fallback: provide final answer based on available information
        return force_final_answer(main_question, subquestion_results)


def force_final_answer(main_question: str, subquestion_results) -> Dict[str, Any]:
    """Create a final answer when forced to conclude"""
    if not subquestion_results:
        final_answer = "Tôi không có đủ thông tin để trả lời câu hỏi này một cách toàn diện."
    else:
        # Combine all answers
        answers = []
        for result in subquestion_results:
            answers.append(f"Về '{result.subquestion}': {result.answer}")
        
        final_answer = f"Dựa trên thông tin có sẵn:\n\n" + "\n\n".join(answers)
        final_answer += f"\n\nĐây là câu trả lời toàn diện nhất tôi có thể cung cấp cho: {main_question}"
    
    return {
        "final_answer": final_answer,
        "is_complete": True
    }