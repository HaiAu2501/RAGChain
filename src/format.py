from pydantic import BaseModel, Field
from typing import List, Optional


class DecomposerFormat(BaseModel):
    """Format for Decomposer Node output"""
    thoughts: str = Field(
        default="",
        description="Lý luận về cách phân tách câu hỏi chính"
    )
    subquestions: List[str] = Field(
        default_factory=list,
        description="Danh sách các câu hỏi phụ được tách ra từ câu hỏi chính"
    )


class ActionFormat(BaseModel):
    """Format for Action Node output"""
    thoughts: str = Field(
        default="",
        description="Lý luận về những truy vấn cần thiết cho câu hỏi phụ này"
    )
    database_queries: List[str] = Field(
        default_factory=list,
        description="Các truy vấn ngôn ngữ tự nhiên cho tìm kiếm ngữ nghĩa trong cơ sở dữ liệu vector (KHÔNG phải truy vấn SQL)"
    )
    web_search_queries: List[str] = Field(
        default_factory=list,
        description="Các truy vấn được tối ưu hóa cho công cụ tìm kiếm web"
    )


class EvaluatorFormat(BaseModel):
    """Format for Evaluator Node output"""
    thoughts: str = Field(
        default="",
        description="Lý luận về chất lượng và mức độ phù hợp của câu trả lời"
    )
    answer: str = Field(
        default="",
        description="Câu trả lời cho câu hỏi phụ dựa trên thông tin đã thu thập"
    )
    confidence: float = Field(
        default=0.0,
        description="Điểm tin cậy về chất lượng câu trả lời (0.0 đến 1.0)",
        ge=0.0,
        le=1.0
    )
    support: float = Field(
        default=0.0,
        description="Mức độ hỗ trợ của câu hỏi phụ này trong việc trả lời câu hỏi chính (0.0 đến 1.0)",
        ge=0.0,
        le=1.0
    )


class MergerFormat(BaseModel):
    """Format for Merger Node output"""
    thoughts: str = Field(
        default="",
        description="Lý luận về việc có đủ thông tin để đưa ra câu trả lời cuối cùng hay không"
    )
    has_final_answer: bool = Field(
        default=False,
        description="Liệu có đủ thông tin để cung cấp câu trả lời cuối cùng hay không"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Câu trả lời cuối cùng cho câu hỏi chính nếu có sẵn"
    )
    selected_subquestion: Optional[str] = Field(
        default=None,
        description="Câu hỏi phụ được chọn để phân tách thêm nếu chưa sẵn sàng đưa ra câu trả lời cuối cùng"
    )
    new_subquestions: List[str] = Field(
        default_factory=list,
        description="Các câu hỏi phụ mới được tạo ra từ câu hỏi phụ đã chọn"
    )