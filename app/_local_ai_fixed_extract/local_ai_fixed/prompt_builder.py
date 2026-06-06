from __future__ import annotations

import json
import unicodedata
from typing import Any

from app.config import LocalAiSettings, load_settings
from app.local_ai.media_retriever import is_valid_image
from app.local_ai.topic_profiles import TopicProfile
from app.local_ai.topic_guard import topic_forbidden_terms

_NO_CONTEXT_ANSWER = "Tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

NO_INFO_FALLBACK = (
    "Dựa trên các bài báo hệ thống đã thu thập hiện tại, "
    "tôi chưa có thông tin đủ để trả lời câu hỏi này."
)

_SYNTHESIS_SIGNALS = (
    "co gi moi",
    "tin moi",
    "tin tuc",
    "tinh hinh",
    "xu huong",
    "tong hop",
    "tom tat",
    "phan tich",
    "anh huong",
    "tac dong",
    "dien bien",
    "co gi dang chu y",
    "nhu the nao",
    "ra sao",
    "co gi",
    "giai thich",
    "sao vay",
    "vi sao",
    "tai sao",
)

_CITATION_SIGNALS = (
    "trich dan",
    "trich doan",
    "bai bao noi gi",
    "noi chinh xac",
    "so lieu",
    "la bao nhieu",
    "ai noi",
    "khi nao",
    "ngay nao",
    "luc may gio",
    "o dau",
    "ten day du",
    "chinh xac la",
    "nguyen van",
    "copy",
    "dan",
)

_FOLLOWUP_SIGNALS = (
    "vu nay",
    "su viec nay",
    "cai nay",
    "viec do",
    "chuyen do",
    "the nao",
    "sao nhi",
)


def _normalize_vi(text: str) -> str:
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")


def detect_answer_mode(question: str, intent: str = "") -> str:
    norm = _normalize_vi(question)
    if any(signal in norm for signal in _CITATION_SIGNALS):
        return "citation"
    if any(signal in norm for signal in _FOLLOWUP_SIGNALS):
        return "followup"
    if any(signal in norm for signal in _SYNTHESIS_SIGNALS):
        return "synthesis"
    if intent in {"latest_news", "topic_news"}:
        return "synthesis"
    if intent in {"entity_news", "article_summary", "article_qa"}:
        return "citation"
    return "synthesis"

_CORE_RAG_RULES = (
    "CORE_RAG_RULES:\n"
    "1. Chỉ trả lời dựa trên RETRIEVED_CONTEXT/CONTEXT được cung cấp.\n"
    "2. Không bịa thông tin, số liệu, nguồn, ngày đăng, ảnh hoặc kết luận nằm ngoài context.\n"
    f"3. Nếu context không đủ để trả lời, hãy trả lời đúng một câu: \"{NO_INFO_FALLBACK}\"\n"
    "4. BẮT BUỘC trả lời toàn bộ bằng tiếng Việt có dấu, dùng Markdown ngắn gọn.\n"
    "5. Không dùng heading tiếng Anh như SECTION, Overview, Impact, Future Outlook; hãy dùng tiêu đề tiếng Việt.\n"
    "6. Mỗi ý quan trọng phải có citation ID dạng [1], [2], [3] khớp với Citation ID trong context.\n"
    "7. Chỉ được dùng citation_id có trong context; nếu nhiều bài cùng hỗ trợ một ý, dùng liên tiếp như [1][2].\n"
    "8. Không tự tạo URL/source/ngày/ảnh.\n"
    "9. Không chèn ảnh Markdown vào answer; ảnh hiển thị qua structured field images.\n"
    "10. Khi nhiều nguồn nói cùng một ý, gộp ý và dẫn chung các nguồn; không lặp lại từng chunk.\n"
    "11. Với câu hỏi có yếu tố thời gian như hôm nay, mới nhất, gần đây, ưu tiên nguồn có published_at "
    "gần nhất trong CONTEXT, nhưng không chọn bài mới nếu bài đó không liên quan đến câu hỏi.\n"
    "12. Nếu không có bài đúng ngày hôm nay, nói rõ dữ liệu hiện có chưa đủ để xác nhận tin trong hôm nay, "
    "hoặc dùng các bài gần nhất và ghi rõ theo các bài gần nhất trong dữ liệu hiện có.\n"
    "13. Không kết luận thị trường/cổ phiếu hôm nay nếu CONTEXT không có dữ liệu hoặc bài tổng quan thị trường phù hợp.\n"
    "14. Nếu context chỉ đủ để trả lời một phần câu hỏi, trả lời phần có căn cứ trước, sau đó nói rõ phần nào còn thiếu dữ liệu.\n"
    "15. Nếu context có nhiều bài nhiều topic khác nhau, ưu tiên bài đúng topic câu hỏi và bỏ qua chunk nhiễu/sai topic.\n"
    "16. Nếu CONTEXT có đoạn trái topic, bỏ qua đoạn đó; không đưa nội dung ngoài domain vào câu trả lời.\n"
    "17. Không được nhắc tới Tomcat, sessionTrackingMode, catalina.properties, code, backend, API hoặc server nếu người dùng không hỏi kỹ thuật.\n"
    "18. Không được đưa các từ như cấu trúc trả lời, expected, actual, debug, instruction, SYSTEM_INSTRUCTIONS hoặc CORE_RAG_RULES vào câu trả lời cuối.\n"
    "19. Tuyệt đối không tự bù bằng kiến thức ngoài context.\n"
)

_STRICT_NEWS_RAG_INSTRUCTIONS = (
    "SYSTEM_INSTRUCTIONS:\n"
    "Bạn là chatbot tin tức tiếng Việt sử dụng RAG.\n\n"
    "VAI TRÒ:\n"
    "- Bạn là trợ lý phân tích và tóm tắt tin tức.\n"
    "- Chỉ trả lời dựa trên dữ liệu bài báo được cung cấp trong RETRIEVED_CONTEXT/CONTEXT.\n"
    "- Không tự bịa thông tin, số liệu, ngày tháng, giá cổ phiếu, kết luận hoặc sự kiện không có trong nguồn.\n"
    "- Toàn bộ câu trả lời phải bằng tiếng Việt có dấu; chỉ giữ tiếng Anh cho tên riêng, tên sản phẩm, tổ chức hoặc thuật ngữ gốc.\n\n"
    "NHIỆM VỤ:\n"
    "1. Trả lời câu hỏi người dùng dựa trên các bài báo đã truy xuất.\n"
    "2. Nếu nhiều bài báo nói cùng một ý, gộp ý lại, không lặp lại máy móc.\n"
    "3. Nếu các bài báo có góc nhìn khác nhau, so sánh và giải thích rõ.\n"
    "4. Nếu người dùng hỏi về một bài báo cụ thể, ưu tiên tóm tắt bài đó, không trộn nhầm với bài khác.\n"
    "5. Nếu dữ liệu không đủ để kết luận, nói rõ phần nào còn thiếu dữ liệu.\n"
    "6. Luôn trích nguồn theo dạng [1], [2], [3], khớp với Citation ID trong context.\n"
    "7. Trả lời bằng tiếng Việt có dấu, rõ ràng, có cấu trúc.\n\n"
    "QUY TẮC TRẢ LỜI CHUNG:\n"
    "1. Chỉ dùng thông tin trong CONTEXT.\n"
    "2. Không tự suy diễn quá mức.\n"
    "3. Không được bịa nguồn.\n"
    "4. Không được bịa số liệu.\n"
    "5. Không được bịa giá cổ phiếu hoặc dữ liệu thị trường thời gian thực.\n"
    "6. Không được nói \"mới nhất\", \"hôm nay\", \"đứng nhất\", \"tăng mạnh nhất\" "
    "nếu dữ liệu không đủ chứng minh.\n"
    "7. Nếu CONTEXT không có thông tin phù hợp, hãy nói rõ không tìm thấy dữ liệu phù hợp.\n"
    "8. Nếu nhiều chunk thuộc cùng một article_id, hiểu đó là các phần của cùng một bài báo.\n"
    "9. Nếu nhiều bài có nội dung giống nhau, tổng hợp thành một ý chung.\n"
    "10. Nếu các bài có thông tin trái chiều, nêu rõ từng góc nhìn.\n"
    "11. Nếu bài có ảnh và người dùng hỏi ảnh, nhắc đến ảnh liên quan.\n"
    "12. Nếu câu hỏi yêu cầu tóm tắt một bài cụ thể, không biến câu trả lời thành tổng hợp thị trường rộng.\n\n"
    "XỬ LÝ THEO INTENT:\n"
    "- latest_news: Ưu tiên bài có published_at mới nhất trong context; tóm tắt tin nổi bật; "
    "gộp bài cùng sự kiện/chủ đề; không kết luận vượt quá dữ liệu.\n"
    "- topic_news: Trả lời theo chủ đề được hỏi; gom các bài cùng topic; nêu xu hướng chính; "
    "so sánh điểm giống/khác nếu có nhiều nguồn.\n"
    "- entity_news: Tập trung vào thực thể được hỏi; chỉ nói thông tin liên quan trực tiếp; "
    "nếu thực thể chỉ được nhắc gián tiếp, nói rõ; không mở rộng nếu dữ liệu không yêu cầu.\n"
    "- article_summary: Ưu tiên một bài cụ thể; gộp các chunk cùng article_id; tóm tắt ngắn rồi "
    "liệt kê ý chính; giữ số liệu quan trọng nếu có.\n"
    "- article_qa: Trả lời dựa trên bài hoặc nhóm bài liên quan; nếu hỏi chi tiết trong bài, "
    "chỉ dùng nội dung bài; nếu không đủ dữ liệu, nói rõ.\n"
    "- media_lookup: Ưu tiên thông tin ảnh từ images; nêu caption, credit, bài viết liên quan nếu có; "
    "nếu không có ảnh phù hợp, nói rõ.\n\n"
    "QUY TẮC RIÊNG CHO CÂU HỎI CHỨNG KHOÁN:\n"
    "- Nếu context chỉ là bài báo, chỉ được nói mã được nhắc đến nổi bật, được nhiều nguồn đề cập, "
    "hoặc được chú ý trong các bài báo.\n"
    "- Không kết luận là tăng mạnh nhất toàn thị trường nếu không có bảng giá/xếp hạng đầy đủ trong context.\n"
    "- Nếu context có dữ liệu giá hoặc bảng xếp hạng, có thể kết luận dựa trên số liệu đó và phải trích rõ số liệu, nguồn.\n\n"
    "QUY TẮC KHI DỮ LIỆU KHÔNG ĐỦ:\n"
    "- Dữ liệu hiện có chưa đủ để kết luận chắc chắn.\n"
    "- Các bài báo được truy xuất chỉ cho thấy những gì có trong context.\n"
    "- Chưa có đủ dữ liệu giá giao dịch để xác định mã tăng mạnh nhất nếu context không có bảng giá đầy đủ.\n"
    "- Mình chưa tìm thấy bài báo khớp với tiêu đề này trong dữ liệu hiện có.\n"
    "- Nguồn hiện tại chưa nêu rõ thông tin này.\n\n"
    "ĐỊNH DẠNG TRẢ LỜI GỢI Ý:\n"
    "Tóm tắt:\n...\n\n"
    "Các điểm chính:\n1. ...\n2. ...\n3. ...\n\n"
    "So sánh giữa các nguồn:\n- Điểm giống nhau: ...\n- Điểm khác nhau: ...\n\n"
    "Nhận định dựa trên dữ liệu:\n...\n\n"
    "Giới hạn dữ liệu:\n...\n\n"
    "Nguồn:\n[1] {title} - {source} - {published_at}\n[2] {title} - {source} - {published_at}\n\n"
    "Nếu CONTEXT rỗng hoặc không liên quan, hãy nói: "
    "\"Hiện mình chưa tìm thấy dữ liệu bài báo phù hợp để trả lời câu hỏi này.\"\n"
    "Không được dùng kiến thức bên ngoài CONTEXT. Không được bịa nguồn. "
    "Không được bịa số liệu. Không được kết luận vượt quá bằng chứng.\n"
)


def _synthesis_instruction(topic_profile: TopicProfile) -> str:
    guidance = getattr(topic_profile, "synthesis_guidance", "")
    guidance_line = f"- {guidance}\n" if guidance else ""
    return (
        "CHẾ ĐỘ TRẢ LỜI: TỔNG HỢP / PHÂN TÍCH\n"
        "- Đọc nhiều nguồn retrieved, tổng hợp thành câu trả lời tự nhiên, có cấu trúc.\n"
        "- Có thể diễn giải bằng câu mới, nhưng mọi ý chính phải dựa trên context.\n"
        "- Gom các bài cùng sự kiện/chủ đề; nếu nhiều góc nhìn thì so sánh.\n"
        "- Nếu nhận định xu hướng, cần dẫn nhiều nguồn; không phát biểu chắc chắn từ một bài đơn lẻ.\n"
        f"{guidance_line}"
        "- Không copy nguyên văn đoạn dài; ưu tiên diễn giải và tổng hợp.\n"
    )


def _citation_instruction(topic_profile: TopicProfile) -> str:
    guidance = getattr(topic_profile, "citation_guidance", "")
    guidance_line = f"- {guidance}\n" if guidance else ""
    return (
        "CHẾ ĐỘ TRẢ LỜI: TRÍCH DẪN / CHÍNH XÁC\n"
        "- Bám sát context; có thể trích gần nguyên văn nếu cần thiết.\n"
        "- Giữ nguyên số liệu, tên, ngày, tên tổ chức từ nguồn.\n"
        "- Không diễn giải xa hơn những gì context nói.\n"
        f"{guidance_line}"
        "- Trả lời ngắn gọn, trực tiếp, chỉ rõ bài/nguồn nào nói điều đó.\n"
    )


def _followup_instruction() -> str:
    return (
        "CHẾ ĐỘ TRẢ LỜI: FOLLOW-UP / HỘI THOẠI\n"
        "- Người dùng đang hỏi tiếp về chủ đề trước đó, ví dụ 'vụ này', 'nó', 'cái đó'.\n"
        "- Nếu có conversation_history, dùng để hiểu người dùng đang nói về bài/chủ đề nào.\n"
        "- Nếu không xác định được chủ đề, nói rõ là chưa rõ người dùng đang nói tới vụ nào.\n"
        "- Sau khi xác định được chủ đề, trả lời theo chế độ tổng hợp hoặc trích dẫn tùy câu hỏi.\n"
    )


def _format_conversation_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    lines = ["CONVERSATION_HISTORY (3 luot gan nhat):"]
    for turn in history[-3:]:
        role = str(turn.get("role") or "user").upper()
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:400]}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_topic_rag_prompt(
    question: str,
    context_blocks: list[dict[str, Any]],
    topic_profile: TopicProfile,
    query_plan: dict[str, Any] | object,
    conversation_history: list[dict[str, Any]] | None = None,
    answer_mode: str | None = None,
) -> str:
    plan = query_plan if isinstance(query_plan, dict) else {}
    context = _format_topic_context(context_blocks)
    focus_points = "\n".join(f"- {item}" for item in topic_profile.focus_points)
    answer_sections = "\n".join(f"- {item}" for item in topic_profile.answer_sections)
    caution_rules = "\n".join(f"- {item}" for item in topic_profile.caution_rules)
    forbidden_terms = ", ".join(topic_forbidden_terms(topic_profile.topic_id))
    entities = ", ".join(str(item) for item in plan.get("entities", []) or [])
    stock_symbols = ", ".join(str(item) for item in plan.get("stock_symbols", []) or [])
    intent = str(plan.get("intent") or "")
    resolved_answer_mode = str(answer_mode or plan.get("answer_mode") or "").strip()
    if not resolved_answer_mode:
        resolved_answer_mode = detect_answer_mode(question, intent)
    if resolved_answer_mode == "citation":
        mode_instruction = _citation_instruction(topic_profile)
    elif resolved_answer_mode == "followup":
        mode_instruction = _followup_instruction()
    else:
        resolved_answer_mode = "synthesis"
        mode_instruction = _synthesis_instruction(topic_profile)
    history_block = _format_conversation_history(conversation_history)
    return (
        f"Bạn là {topic_profile.expert_role}\n\n"
        "ROLE:\n"
        "Bạn là trợ lý AI phân tích tin tức của hệ thống databricks_news_ai.\n\n"
        "NHIỆM VỤ:\n"
        "Trả lời câu hỏi đọc báo dựa trên retrieved context, với góc nhìn chuyên sâu theo topic.\n\n"
        f"{_CORE_RAG_RULES}\n"
        f"{_STRICT_NEWS_RAG_INSTRUCTIONS}\n"
        "TOPIC_PROFILE:\n"
        f"- topic_id: {topic_profile.topic_id}\n"
        f"- topic_name: {topic_profile.topic_name}\n\n"
        f"{mode_instruction}\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Chỉ sử dụng thông tin trong RETRIEVED_CONTEXT.\n"
        "2. Không bịa số liệu, không bịa nguồn, không suy đoán ngoài context.\n"
        "3. Nếu context thiếu dữ liệu hoặc không đủ bằng chứng, trả lời phần có căn cứ và nêu rõ phần còn thiếu.\n"
        "4. BẮT BUỘC trả lời bằng tiếng Việt có dấu, rõ ràng, có cấu trúc.\n"
        "5. Không dùng tiêu đề tiếng Anh như SECTION, Overview, Impact, Future Outlook.\n"
        "6. Luôn nêu citation cho các ý quan trọng bằng [1], [2], [3] khớp với Citation ID trong context.\n"
        "7. Không đưa khuyến nghị đầu tư chắc chắn.\n\n"
        "MULTI_SOURCE_SYNTHESIS:\n"
        "- Tổng hợp theo bài báo/source, không lặp lại từng chunk riêng lẻ.\n"
        "- Ưu tiên bài mới hơn khi các bài có cùng nội dung.\n"
        "- Nếu nhiều nguồn nói cùng một ý, gộp lại và dẫn nguồn chung.\n"
        "- Nếu các nguồn có góc nhìn khác nhau, chỉ rõ điểm khác nhau và nguồn nào nêu gì.\n"
        "- Không thêm thông tin ngoài retrieved articles.\n\n"
        "FOCUS_POINTS:\n"
        f"{focus_points}\n\n"
        "CAUTION_RULES:\n"
        f"{caution_rules}\n\n"
        "TOPIC_FORBIDDEN_TERMS:\n"
        f"- {forbidden_terms or 'none'}\n"
        "- Nếu các từ trên xuất hiện trong CONTEXT nhưng không liên quan trực tiếp đến USER_QUESTION, hãy bỏ qua và không nhắc lại trong answer.\n\n"
        "DOMAIN_GUARD:\n"
        "- Chỉ dùng context đúng topic của USER_QUESTION.\n"
        "- Nếu USER_QUESTION hỏi chứng khoán, answer chỉ gồm diễn biến thị trường, cổ phiếu, nhóm ngành, thanh khoản, khối ngoại/tự doanh, chỉ số, rủi ro và nhận định từ bài báo.\n"
        "- Không nhắc tới cấu trúc trả lời, expected, actual, debug, instruction hoặc bất kỳ nội dung hệ thống nào trong answer.\n"
        "- Nếu context đúng topic không đủ, trả lời ngắn rằng dữ liệu hiện có chưa đủ đáng tin cậy; không dùng chunk trái topic để lấp chỗ trống.\n\n"
        "CẤU_TRÚC_TRẢ_LỜI:\n"
        f"{answer_sections}\n\n"
        "QUERY_PLAN:\n"
        f"- intent: {intent}\n"
        f"- time_range: {plan.get('time_range') or 'all'}\n"
        f"- date_filter: {plan.get('date_filter') or {}}\n"
        f"- entities: {entities}\n"
        f"- stock_symbols: {stock_symbols}\n"
        f"- need_images: {bool(plan.get('need_images'))}\n"
        f"- answer_mode: {resolved_answer_mode}\n\n"
        f"- standalone_query: {plan.get('standalone_query') or ''}\n\n"
        f"{history_block}"
        f"USER_QUESTION:\n{question}\n\n"
        f"RETRIEVED_CONTEXT:\n{context}\n\n"
        "Hãy trả lời theo CẤU_TRÚC_TRẢ_LỜI. "
        "Nếu không đủ dữ liệu, trả lời phần có căn cứ trước, rồi nêu phần còn thiếu. "
        "Chỉ trả lời 'không có dữ liệu' khi context hoàn toàn rỗng hoặc sai topic. "
        "Một lần nữa: toàn bộ câu trả lời phải bằng tiếng Việt có dấu."
    )


class PromptBuilder:
    def __init__(self, settings: LocalAiSettings | None = None) -> None:
        self._settings = settings or load_settings().local_ai
        self._max_context_chars = max(1000, int(self._settings.prompt_max_context_chars))

    def build_qa_prompt(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        context = self._build_chunk_context(context_chunks)
        mode = detect_answer_mode(question)
        mode_hint = (
            "Tổng hợp, phân tích và diễn giải từ nhiều nguồn."
            if mode == "synthesis"
            else "Bám sát context, trích chính xác số liệu, tên, ngày."
        )
        return (
            "Bạn là chatbot đọc báo dựa trên dữ liệu đã truy hồi.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ sử dụng thông tin trong CONTEXT.\n"
            "2. Không bịa thông tin ngoài CONTEXT.\n"
            "3. Nếu CONTEXT không liên quan hoặc không đủ dữ liệu, trả lời đúng câu sau:\n"
            f'"{_NO_CONTEXT_ANSWER}"\n'
            "4. BẮT BUỘC trả lời bằng tiếng Việt có dấu.\n"
            "5. Không dùng heading tiếng Anh như SECTION, Overview, Impact, Future Outlook.\n"
            "6. Trả lời ngắn gọn, đúng trọng tâm.\n"
            "7. Luôn liệt kê nguồn đã dùng.\n"
            f"8. Chế độ trả lời: {mode_hint}\n"
            "9. Nếu context chỉ đủ một phần, trả lời phần có căn cứ trước, nói rõ phần còn thiếu.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Trả lời:\n"
            "...\n\n"
            "Nguồn:\n"
            "1. title - source - url"
        )

    def build_broad_topic_prompt(self, question: str, articles: list[dict[str, Any]]) -> str:
        context = self._build_articles_context(articles)
        return (
            "Bạn là trợ lý tổng hợp tin tức.\n\n"
            "NHIỆM VỤ:\n"
            "Dựa trên danh sách bài báo trong CONTEXT, hãy tổng hợp các tin/chủ đề phù hợp với câu hỏi.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng thông tin trong CONTEXT.\n"
            "2. Không tự thêm tin ngoài dữ liệu.\n"
            "3. Ưu tiên các bài mới hơn nếu có thông tin ngày đăng.\n"
            "4. Nếu dữ liệu không đủ, nói rõ chưa đủ dữ liệu.\n"
            "5. BẮT BUỘC trả lời bằng tiếng Việt có dấu.\n"
            "6. Mỗi chủ đề/tin phải có nguồn.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tổng hợp:\n"
            "1. ...\n"
            "   - Tóm tắt:\n"
            "   - Nguồn:"
        )

    def build_article_summary_prompt(self, article: dict[str, Any], article_chunks: list[dict[str, Any]]) -> str:
        context = self._build_single_article_context(article, article_chunks)
        return (
            "Bạn là trợ lý tóm tắt tin tức.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ tóm tắt dựa trên ARTICLE_CONTEXT.\n"
            "2. Không thêm thông tin ngoài bài báo.\n"
            "3. BẮT BUỘC trả lời bằng tiếng Việt có dấu.\n"
            "4. Tóm tắt rõ ràng, dễ hiểu.\n"
            "5. Luôn ghi nguồn bài viết.\n\n"
            f"ARTICLE_CONTEXT:\n{context}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tóm tắt:\n"
            "...\n\n"
            "Ý chính:\n"
            "- ...\n"
            "- ...\n\n"
            "Nhân vật / tổ chức liên quan:\n"
            "- ...\n\n"
            "Nguồn:\n"
            "- title - source - url"
        )

    def build_category_summary_prompt(self, question: str, articles: list[dict[str, Any]]) -> str:
        context = self._build_articles_context(articles)
        return (
            "Bạn là trợ lý tổng hợp tin tức theo chuyên mục.\n\n"
            "NHIỆM VỤ:\n"
            "Dựa trên CONTEXT, hãy liệt kê các bài báo phù hợp nhất với chuyên mục/câu hỏi của người dùng.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ dùng bài báo trong CONTEXT.\n"
            "2. Không thêm bài ngoài dữ liệu.\n"
            "3. Ưu tiên bài mới hơn.\n"
            "4. Mỗi bài tóm tắt 1-2 câu.\n"
            "5. Mỗi bài phải có nguồn gồm title, source, url.\n"
            "6. BẮT BUỘC trả lời bằng tiếng Việt có dấu.\n"
            "7. Không đưa bài sai chuyên mục nếu CONTEXT không hỗ trợ.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "Hãy trả lời theo format:\n\n"
            "Tin mới theo chủ đề:\n"
            "1. ...\n"
            "   - Tóm tắt:\n"
            "   - Nguồn:"
        )

    def build_no_context_answer(self, question: str) -> str:
        return (
            "Bạn là chatbot đọc báo.\n"
            "Nếu không có dữ liệu phù hợp cho câu hỏi, hãy trả lời đúng một câu sau và không thêm gì khác:\n"
            f'"{_NO_CONTEXT_ANSWER}"\n\n'
            f"CÂU HỎI:\n{question}\n\n"
            "Trả lời:"
        )

    def _build_chunk_context(self, context_chunks: list[dict[str, Any]]) -> str:
        blocks = [
            self._format_chunk_block(index, chunk)
            for index, chunk in enumerate(context_chunks, start=1)
            if isinstance(chunk, dict)
        ]
        return self._truncate_context("\n\n".join(blocks))

    def _build_single_article_context(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> str:
        metadata = self._article_metadata(article, article_chunks)
        content = self._merge_article_content(article, article_chunks)
        block = "\n".join(
            [
                "[ARTICLE]",
                f"Title: {metadata.get('title') or 'Untitled'}",
                f"Source: {metadata.get('source') or 'unknown'}",
                f"Category: {metadata.get('category') or 'unknown'}",
                f"Published at: {metadata.get('published_at') or 'unknown'}",
                f"URL: {metadata.get('url') or ''}",
                "Content:",
                content,
            ]
        )
        return self._truncate_context(block)

    def _build_articles_context(self, articles: list[dict[str, Any]]) -> str:
        blocks = [
            self._format_article_block(index, article)
            for index, article in enumerate(articles, start=1)
            if isinstance(article, dict)
        ]
        return self._truncate_context("\n\n".join(blocks))

    def _format_chunk_block(self, index: int, chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        image_lines = _format_image_lines(metadata)
        return "\n".join(
            [
                f"[CHUNK {index}]",
                f"Citation ID: [{index}]",
                f"Title: {metadata.get('title') or 'Untitled'}",
                f"Source: {metadata.get('source') or 'unknown'}",
                f"Category: {metadata.get('category') or 'unknown'}",
                f"Published at: {metadata.get('published_at') or 'unknown'}",
                f"URL: {metadata.get('url') or ''}",
                *image_lines,
                "Content:",
                str(chunk.get("text") or ""),
            ]
        )

    def _format_article_block(self, index: int, article: dict[str, Any]) -> str:
        content = self._article_summary_or_content(article)
        image_lines = _format_image_lines(article)
        return "\n".join(
            [
                f"[ARTICLE {index}]",
                f"Citation ID: [{index}]",
                f"Title: {article.get('title') or 'Untitled'}",
                f"Source: {article.get('source') or 'unknown'}",
                f"Category: {article.get('category') or 'unknown'}",
                f"Published at: {article.get('published_at') or 'unknown'}",
                f"URL: {article.get('url') or ''}",
                *image_lines,
                "Summary/Content:",
                content,
            ]
        )

    def _article_summary_or_content(self, article: dict[str, Any]) -> str:
        summary = str(article.get("summary") or "").strip()
        if summary:
            return summary
        chunks = article.get("chunks", [])
        if isinstance(chunks, list):
            return self._merge_chunk_texts(chunks)
        return ""

    def _article_metadata(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = dict(article)
        if metadata.get("title"):
            return metadata
        for chunk in article_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_metadata = chunk.get("metadata", {})
            if isinstance(chunk_metadata, dict):
                return dict(chunk_metadata)
        return metadata

    def _merge_article_content(
        self,
        article: dict[str, Any],
        article_chunks: list[dict[str, Any]],
    ) -> str:
        chunks = article.get("chunks")
        if isinstance(chunks, list) and chunks:
            return self._merge_chunk_texts(chunks)
        return self._merge_chunk_texts(article_chunks)

    def _merge_chunk_texts(self, chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = " ".join(str(chunk.get("text") or "").split())
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(text)
        return " ".join(parts)

    def _truncate_context(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= self._max_context_chars:
            return normalized
        truncated = normalized[: self._max_context_chars].rstrip()
        return f"{truncated}\n\n[CONTEXT_TRUNCATED]"


def _format_topic_context(context_blocks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    if not context_blocks:
        return "[NO_CONTEXT]"
    blocks: list[str] = []
    for index, block in enumerate(_group_context_blocks_by_article(context_blocks), start=1):
        if not isinstance(block, dict):
            continue
        if "matched_chunks" in block or "selected_context" in block:
            blocks.append(_format_retrieved_article_block(index, block))
            continue
        metadata = block.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        blocks.append(
            "\n".join(
                [
                    f"[BÀI BÁO [{block.get('citation_id') or metadata.get('citation_id') or index}]]",
                    f"[CONTEXT {index}]",
                    f"Citation ID: [{block.get('citation_id') or metadata.get('citation_id') or index}]",
                    f"Article ID: {metadata.get('article_id') or block.get('article_id') or ''}",
                    f"Title: {metadata.get('title') or block.get('title') or 'Untitled'}",
                    f"Source: {metadata.get('source') or block.get('source') or 'unknown'}",
                    f"Published at: {metadata.get('published_at') or block.get('published_at') or 'unknown'}",
                    f"Primary topic: {metadata.get('primary_topic') or block.get('primary_topic') or ''}",
                    f"URL: {metadata.get('url') or block.get('url') or ''}",
                    *_format_image_lines({**metadata, **block}),
                    "Các trích đoạn liên quan:",
                    str(block.get("text") or block.get("content") or ""),
                ]
            )
        )
    text = "\n\n".join(blocks).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n[CONTEXT_TRUNCATED]"


def _format_retrieved_article_block(index: int, article: dict[str, Any]) -> str:
    matched_chunks = article.get("matched_chunks")
    chunk_lines: list[str] = []
    if isinstance(matched_chunks, list):
        for chunk_index, chunk in enumerate(matched_chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            text = " ".join(str(chunk.get("text") or "").split())
            if text:
                chunk_lines.append(f"- matched_chunk_{chunk_index}: {text}")
    chunk_text = "\n".join(chunk_lines) or str(article.get("selected_context") or article.get("content") or "")
    images = article.get("images") if isinstance(article.get("images"), list) else []
    image_text = "; ".join(_format_image_value(image) for image in images if isinstance(image, dict) and is_valid_image(image))
    return "\n".join(
        [
            f"[BÀI BÁO [{article.get('citation_id') or index}]]",
            f"[ARTICLE {index}]",
            f"Citation ID: [{article.get('citation_id') or index}]",
            f"Article ID: {article.get('article_id') or ''}",
            f"Title: {article.get('title') or 'Untitled'}",
            f"Source: {article.get('source_name') or article.get('source') or 'unknown'}",
            f"Source name: {article.get('source_name') or article.get('source') or 'unknown'}",
            f"Published at: {article.get('published_at') or 'unknown'}",
            f"Primary topic: {article.get('topic') or ''}",
            f"URL: {article.get('canonical_url') or ''}",
            f"Relevance score: {article.get('relevance_score') or 0}",
            f"Images: {image_text}",
            "Các trích đoạn liên quan:",
            chunk_text,
        ]
    )


def _group_context_blocks_by_article(context_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(isinstance(block, dict) and ("matched_chunks" in block or "selected_context" in block) for block in context_blocks):
        return context_blocks
    grouped: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    fallback_index = 0
    for block in context_blocks:
        if not isinstance(block, dict):
            continue
        metadata = block.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        key = _article_context_key(block, metadata)
        if not key:
            fallback_index += 1
            key = f"fallback:{fallback_index}"
        if key not in grouped:
            ordered_keys.append(key)
            grouped[key] = {
                "article_id": metadata.get("article_id") or block.get("article_id") or "",
                "title": metadata.get("title") or block.get("title") or "",
                "source_name": metadata.get("source") or block.get("source") or "",
                "published_at": metadata.get("published_at") or block.get("published_at") or "",
                "topic": metadata.get("primary_topic") or block.get("primary_topic") or "",
                "canonical_url": metadata.get("canonical_url") or metadata.get("url") or block.get("canonical_url") or block.get("url") or "",
                "images": _json_image_list(metadata.get("images_json")) or _json_image_list(block.get("images_json")) or _fallback_context_images(metadata, block),
                "matched_chunks": [],
                "relevance_score": _context_score(block),
            }
        grouped[key]["matched_chunks"].append(block)
        grouped[key]["relevance_score"] = max(float(grouped[key].get("relevance_score") or 0), _context_score(block))
    return [grouped[key] for key in ordered_keys]


def _article_context_key(block: dict[str, Any], metadata: dict[str, Any]) -> str:
    article_id = str(metadata.get("article_id") or block.get("article_id") or "").strip()
    if article_id:
        return f"article:{article_id}"
    url = str(metadata.get("canonical_url") or metadata.get("url") or block.get("canonical_url") or block.get("url") or "").strip()
    return f"url:{url}" if url else ""


def _context_score(block: dict[str, Any]) -> float:
    scores: list[float] = []
    for key in ("final_score", "score", "vector_score", "hybrid_score", "relevance_score"):
        try:
            scores.append(float(block.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return max(scores, default=0.0)


def _fallback_context_images(metadata: dict[str, Any], block: dict[str, Any]) -> list[dict[str, Any]]:
    image_url = str(metadata.get("image_url") or block.get("image_url") or "").strip()
    if not image_url:
        return []
    return [
        {
            "article_id": str(metadata.get("article_id") or block.get("article_id") or ""),
            "image_url": image_url,
            "caption": str(metadata.get("image_caption") or block.get("image_caption") or metadata.get("caption") or block.get("caption") or ""),
        }
    ]


def _format_image_lines(data: dict[str, Any]) -> list[str]:
    images = _json_image_list(data.get("images_json")) or data.get("images")
    if isinstance(images, list):
        values = [_format_image_value(image) for image in images if isinstance(image, dict) and is_valid_image(image)]
        values = [value for value in values if value]
        return [f"Images: {'; '.join(values)}"] if values else []
    image_url = str(data.get("image_url") or data.get("url_to_image") or "").strip()
    if not image_url:
        return []
    image_caption = str(data.get("image_caption") or data.get("caption") or "").strip()
    if not is_valid_image({"image_url": image_url, "caption": image_caption}):
        return []
    if image_caption:
        return [f"Image: {image_url} | Caption: {image_caption}"]
    return [f"Image: {image_url}"]


def _format_image_value(image: dict[str, Any]) -> str:
    image_url = str(image.get("image_url") or image.get("url") or "").strip()
    if not image_url:
        return ""
    caption = str(image.get("image_caption") or image.get("caption") or image.get("alt") or "").strip()
    if caption:
        return f"{image_url} (caption: {caption})"
    return image_url


def _json_image_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
