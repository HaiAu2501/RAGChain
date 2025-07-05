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
        context = "Đây là câu hỏi chính cần được trả lời."
    else:
        # Later iterations - decompose selected subquestion
        # Find the selected subquestion from merger results
        question_to_decompose = state.get("current_subquestion", state["question"])
        context = f"""Đây là một câu hỏi phụ được chọn để phân tách thêm. 
        Câu hỏi chính: {state['main_question']}
        Các câu hỏi phụ và câu trả lời trước đó: {[f'H: {r.subquestion}, Đ: {r.answer}' for r in state.get('subquestion_results', [])]}"""
    
    # Get current time for context
    current_time = datetime.now().strftime("%B %Y")  # e.g., "December 2024"
    current_date = datetime.now().strftime("%Y-%m-%d")  # e.g., "2024-12-01"
    
    system_prompt = f"""Bạn là một chuyên gia phân tách câu hỏi. Nhiệm vụ của bạn là chia những câu hỏi phức tạp thành các câu hỏi phụ đơn giản hơn, tập trung hơn và có thể được trả lời độc lập.

BỐI CẢNH THỜI GIAN HIỆN TẠI:
- Ngày hiện tại: {current_date}
- Tháng/năm hiện tại: {current_time}
- Sử dụng thông tin này khi tạo các câu hỏi phụ liên quan đến thời gian

Hướng dẫn:
1. Tạo ra chính xác tối đa {N_BRANCHES} câu hỏi phụ
2. Mỗi câu hỏi phụ phải cụ thể và có thể trả lời được thông qua tìm kiếm cơ sở dữ liệu hoặc tìm kiếm web
3. Các câu hỏi phụ phải cùng nhau giúp trả lời câu hỏi gốc
4. Tránh các câu hỏi phụ dư thừa hoặc quá giống nhau
5. Tập trung vào những khía cạnh quan trọng nhất cần thiết để trả lời câu hỏi gốc
6. Bao gồm bối cảnh thời gian trong các câu hỏi phụ khi câu hỏi gốc liên quan đến những phát triển gần đây, sự kiện hiện tại hoặc thông tin nhạy cảm về thời gian
7. Đối với câu hỏi về "mới nhất", "gần đây", "hiện tại", hãy đảm bảo các câu hỏi phụ chỉ định khoảng thời gian phù hợp

Ví dụ về câu hỏi phụ nhận biết thời gian:
- "Những phát triển AI mới nhất trong năm 2024 là gì?"
- "Những chính sách khí hậu gần đây nào đã được thực hiện?"
- "Việc áp dụng năng lượng tái tạo đã thay đổi như thế nào trong những năm gần đây?"

Bối cảnh: {context}

Vui lòng trả lời bằng tiếng Việt."""

    human_prompt = f"""Hãy phân tách câu hỏi này thành các câu hỏi phụ tập trung:
    
Câu hỏi: {question_to_decompose}

Xem xét bối cảnh thời gian hiện tại ({current_time}) khi tạo các câu hỏi phụ, đặc biệt nếu câu hỏi liên quan đến những phát triển gần đây, xu hướng hiện tại hoặc thông tin nhạy cảm về thời gian.

Cung cấp lý luận của bạn và sau đó liệt kê các câu hỏi phụ."""

    try:
        response = DECOMPOSER_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Limit to N_BRANCHES subquestions
        subquestions = response.subquestions[:N_BRANCHES]

        print("\n[PHÂN TÁCH] Đang phân tách câu hỏi...")

        print(f"Đã phân tách {len(subquestions)} câu hỏi phụ từ '{question_to_decompose}':")
        for i, subq in enumerate(subquestions):
            print(f"  {i + 1}. {subq}")
        
        return {
            "current_subquestions": subquestions,
            "main_question": state.get("main_question", state["question"]),
            "current_iteration": state.get("current_iteration", 0)
        }
        
    except Exception as e:
        print(f"Lỗi trong decomposer_node: {str(e)}")
        # Fallback: use original question as single subquestion
        return {
            "current_subquestions": [question_to_decompose],
            "main_question": state.get("main_question", state["question"]),
            "current_iteration": state.get("current_iteration", 0)
        }