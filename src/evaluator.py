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
        if source_type == "DATABASE":
            source_type = "CƠ SỞ DỮ LIỆU"
        elif source_type == "WEB":
            source_type = "WEB"
        
        formatted_results.append(f"\n--- TRUY VẤN {source_type}: {result.query} ---")
        
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
    
    system_prompt = """Bạn là một chuyên gia đánh giá thông tin với cách tiếp cận phê phán và nghiêm ngặt. Nhiệm vụ của bạn là phân tích thông tin đã thu thập và cung cấp câu trả lời toàn diện cho một câu hỏi phụ.

HƯỚNG DẪN ĐÁNH GIÁ PHẢN BIỆN:
1. XÁC MINH NGUỒN: Cẩn thận xác minh xem các nguồn có khớp với yêu cầu trong câu hỏi chính không
   - Nếu câu hỏi chỉ định nguồn cụ thể (ví dụ: "theo báo cáo của X và Y"), hãy đảm bảo thông tin đến từ đúng những nguồn đó
   - Phân biệt giữa các ấn phẩm khác nhau và phạm vi bao quát của chúng
   - Đánh dấu khi các nguồn không phù hợp với yêu cầu của câu hỏi

2. TÍNH CHÍNH XÁC THÔNG TIN: Hãy cực kỳ thận trọng về các tuyên bố thực tế
   - Đối chiếu thông tin qua nhiều nguồn
   - Xác định mâu thuẫn hoặc không nhất quán
   - Xác minh các chi tiết cụ thể như tên, ngày tháng, cáo buộc và hoàn cảnh
   - Hãy hoài nghi về những tuyên bố chưa được xác minh

3. TÍNH TOÀN DIỆN CỦA CÂU TRẢ LỜI: Đảm bảo câu trả lời giải quyết trực tiếp câu hỏi phụ
   - Câu trả lời phải cụ thể và chính xác
   - Tránh những phản hồi chung chung hoặc mơ hồ
   - Bao gồm bối cảnh và chi tiết liên quan

4. CHẤM ĐIỂM ĐỘ TIN CẬY (0.0-1.0) - HÃY NGHIÊM NGẶT:
   - 0.9-1.0: Nhiều nguồn chất lượng cao với thông tin nhất quán, đã được xác minh
   - 0.7-0.8: Nguồn tốt với thông tin phần lớn nhất quán, có một số khoảng trống nhỏ
   - 0.5-0.6: Nguồn chất lượng hỗn hợp hoặc có một số không nhất quán
   - 0.3-0.4: Nguồn hạn chế hoặc có sự không nhất quán đáng kể
   - 0.0-0.2: Nguồn kém, mâu thuẫn lớn hoặc thông tin không đủ

5. CHẤM ĐIỂM HỖ TRỢ (0.0-1.0) - Đánh giá mức độ liên quan đến câu hỏi chính:
   - Điều này có giải quyết trực tiếp các yêu cầu của câu hỏi chính không?
   - Thông tin này quan trọng như thế nào đối với câu trả lời hoàn chỉnh?
   - Nó có lấp đầy những khoảng trống quan trọng trong hiểu biết không?

QUAN TRỌNG: Nếu thông tin thu thập được không đáp ứng các yêu cầu cụ thể của câu hỏi chính (ví dụ: sai nguồn, sai người, sai chi tiết), hãy cho điểm tin cậy THẤP và giải thích lý do trong lý luận của bạn.

Vui lòng trả lời bằng tiếng Việt."""

    human_prompt = f"""Vui lòng đánh giá thông tin sau đây và cung cấp câu trả lời cho câu hỏi phụ:

CÂU HỎI CHÍNH: {main_question}

CÂU HỎI PHỤ: {current_subquestion}

THÔNG TIN ĐÃ THU THẬP:
{formatted_results}

Dựa trên thông tin này, vui lòng cung cấp:
1. Lý luận của bạn về chất lượng và mức độ liên quan của thông tin
2. Câu trả lời toàn diện cho câu hỏi phụ
3. Điểm tin cậy (0.0-1.0) cho câu trả lời của bạn
4. Điểm hỗ trợ (0.0-1.0) cho mức độ giúp đỡ trong việc trả lời câu hỏi chính"""

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

        print("\n[ĐÁNH GIÁ] Đang đánh giá câu hỏi phụ...")
        print(f"Đã đánh giá câu hỏi phụ: {current_subquestion}")
        print(f"Câu trả lời: {response.answer}")
        print(f"Độ tin cậy: {response.confidence}, Độ hỗ trợ: {response.support}")
        
        return {
            "subquestion_results": updated_results,
            "current_subquestions": remaining_subquestions
        }
        
    except Exception as e:
        print(f"Lỗi trong evaluator_node: {str(e)}")
        
        # Fallback: create a basic result
        fallback_result = SubquestionResult(
            subquestion=current_subquestion,
            answer=f"Không thể đánh giá thông tin đúng cách do lỗi: {str(e)}",
            confidence=0.1,
            support=0.1,
            thoughts=f"Đã xảy ra lỗi trong quá trình đánh giá: {str(e)}"
        )
        
        existing_results = state.get("subquestion_results", [])
        updated_results = existing_results + [fallback_result]
        remaining_subquestions = state.get("current_subquestions", [])[1:]
        
        return {
            "subquestion_results": updated_results,
            "current_subquestions": remaining_subquestions
        }