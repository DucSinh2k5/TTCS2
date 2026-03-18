import streamlit as st
from groq import Groq
from pathlib import Path


API_KEY_FILE_PATH = Path(__file__).parent / "api_key.txt"


def get_groq_api_key() -> str:
    try:
        api_key = API_KEY_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return api_key

st.title("So sánh 2 xe ô tô")
st.markdown("Nhập tên 2 xe ô tô để AI so sánh chi tiết.")

recommended_df = st.session_state.get("recommended_df")
selected_from_top5: list[str] = []

if recommended_df is not None and not recommended_df.empty and "Ten_xe" in recommended_df.columns:
    st.sidebar.header("⚖️ So sánh từ Top 5 recommend")
    top5_options = recommended_df["Ten_xe"].dropna().unique().tolist()
    selected_from_top5 = st.sidebar.multiselect(
        "Chọn đúng 2 xe từ Top 5",
        options=top5_options,
        default=top5_options[:2] if len(top5_options) >= 2 else top5_options,
        max_selections=2,
    )
    if len(selected_from_top5) != 2:
        st.sidebar.warning("Vui lòng chọn đúng 2 xe để so sánh.")
else:
    st.sidebar.info("Chưa có dữ liệu Top 5. Hãy dự đoán giá xe trước ở trang dự đoán.")

api_key = get_groq_api_key()
client = Groq(api_key=api_key) if api_key else None

col1, col2 = st.columns(2)
with col1:
    xe1 = st.text_input(
        "Xe thứ nhất",
        value=selected_from_top5[0] if len(selected_from_top5) == 2 else "",
        placeholder="VD: Toyota Camry 2024",
    )
with col2:
    xe2 = st.text_input(
        "Xe thứ hai",
        value=selected_from_top5[1] if len(selected_from_top5) == 2 else "",
        placeholder="VD: Honda Accord 2024",
    )

if st.button("So sánh ngay", type="primary"):
    if not xe1 or not xe2:
        st.warning("Vui lòng nhập tên cả 2 xe.")
    elif not api_key:
        st.error("Thiếu Groq API key trong file app/api_key.txt")
    else:
        prompt = f"""Hãy so sánh chi tiết 2 xe ô tô sau: **{xe1}** và **{xe2}** và cho hình ảnh minh họa. Nếu không có hình ảnh tiện, cho 1 đường link chứa ảnh.

Vui lòng so sánh theo các tiêu chí:
1. **Thông số kỹ thuật** (động cơ, công suất, mô-men xoắn, hộp số)
2. **Kích thước & trọng lượng**
3. **Trang bị an toàn**
4. **Tiện nghi & công nghệ**
5. **Mức tiêu hao nhiên liệu**
6. **Giá bán** (tại Việt Nam nếu có)
7. **Ưu & nhược điểm** của từng xe
8. **Kết luận**: Nên chọn xe nào và phù hợp với đối tượng nào?

Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu."""

        def stream_response():
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        try:
            st.write_stream(stream_response())
            # pass
        except Exception as e:
            st.error(f"Lỗi: {e}")


