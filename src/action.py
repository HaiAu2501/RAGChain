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
    
    system_prompt = f"""Bạn là một chuyên gia tạo truy vấn tìm kiếm. Nhiệm vụ của bạn là tạo ra các truy vấn tìm kiếm hiệu quả để tìm thông tin trả lời một câu hỏi phụ cụ thể.

THÔNG TIN QUAN TRỌNG VỀ CƠ SỞ DỮ LIỆU:
- Cơ sở dữ liệu là CƠ SỞ DỮ LIỆU VECTOR (tìm kiếm ngữ nghĩa), KHÔNG phải cơ sở dữ liệu SQL
- Các truy vấn cơ sở dữ liệu phải là các truy vấn NGÔN NGỮ TỰ NHIÊN, không phải lệnh SQL
- Hãy nghĩ về các truy vấn cơ sở dữ liệu như từ khóa hoặc cụm từ để tìm tài liệu liên quan
- Ví dụ về truy vấn cơ sở dữ liệu TỐT: "trí tuệ nhân tạo phát triển gần đây", "ứng dụng học máy 2024", "tác động biến đổi khí hậu nông nghiệp"
- Ví dụ về truy vấn cơ sở dữ liệu XẤU: "SELECT * FROM...", "WHERE year = 2024", bất kỳ cú pháp SQL nào

BỐI CẢNH THỜI GIAN HIỆN TẠI:
- Ngày hiện tại: {current_date}
- Tháng/năm hiện tại: {current_time}
- Sử dụng thông tin này để tạo các truy vấn phù hợp với thời gian khi cần

Hướng dẫn:
1. Tạo ra tối đa {N_QUERIES} TRUY VẤN CƠ SỞ DỮ LIỆU dưới dạng thuật ngữ tìm kiếm ngôn ngữ tự nhiên cho cơ sở dữ liệu vector
2. Tạo ra tối đa {N_QUERIES} TRUY VẤN TÌM KIẾM WEB được tối ưu hóa cho các công cụ tìm kiếm web
3. Các truy vấn cơ sở dữ liệu nên tập trung vào việc tìm tài liệu có khái niệm, từ khóa hoặc chủ đề liên quan
4. Các truy vấn tìm kiếm web nên được tối ưu hóa để tìm thông tin gần đây, sự kiện hiện tại hoặc dữ liệu mới nhất
5. Bao gồm bối cảnh thời gian phù hợp trong các truy vấn khi câu hỏi liên quan đến thông tin gần đây hoặc hiện tại
6. Làm cho các truy vấn đủ cụ thể để tìm thông tin liên quan nhưng không quá hẹp đến mức bỏ lỡ nội dung quan trọng
7. Xem xét các cách diễn đạt hoặc phương pháp khác nhau để có phạm vi bao phủ tốt hơn

Bối cảnh câu hỏi chính: {state.get('main_question', 'Không được cung cấp')}

Vui lòng trả lời bằng tiếng Việt."""

    human_prompt = f"""Tạo các truy vấn tìm kiếm hiệu quả cho câu hỏi phụ này:

Câu hỏi phụ: {subquestion}

Nhớ rằng:
- Truy vấn cơ sở dữ liệu: Thuật ngữ ngôn ngữ tự nhiên cho tìm kiếm ngữ nghĩa (KHÔNG phải SQL)
- Truy vấn web: Được tối ưu hóa cho các công cụ tìm kiếm
- Bao gồm bối cảnh thời gian nếu câu hỏi liên quan đến những phát triển gần đây hoặc thông tin hiện tại

Vui lòng cung cấp lý luận của bạn về thông tin cần thiết và sau đó tạo ra các truy vấn tìm kiếm cơ sở dữ liệu và web phù hợp."""

    try:
        response = ACTION_MODEL.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Limit queries to N_QUERIES
        database_queries = response.database_queries[:N_QUERIES]
        web_queries = response.web_search_queries[:N_QUERIES]

        print("\n[TÁC VỤ] Đang tạo truy vấn cho câu hỏi phụ...")

        print(f"Nút Tác vụ đã tạo {len(database_queries)} truy vấn cơ sở dữ liệu và {len(web_queries)} truy vấn web cho câu hỏi phụ: {subquestion}")
        print("Truy vấn Cơ sở dữ liệu:", database_queries)
        print("Truy vấn Web:", web_queries)

        return {
            "current_thoughts": response.thoughts,
            "current_database_queries": database_queries,
            "current_web_queries": web_queries,
            "current_subquestion": subquestion
        }
        
    except Exception as e:
        print(f"Lỗi trong action_node: {str(e)}")
        # Fallback: create simple queries based on the subquestion
        return {
            "current_thoughts": f"Đã xảy ra lỗi, sử dụng truy vấn dự phòng cho: {subquestion}",
            "current_database_queries": [subquestion],
            "current_web_queries": [subquestion],
            "current_subquestion": subquestion
        }