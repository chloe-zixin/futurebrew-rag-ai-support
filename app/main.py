import streamlit as st
from rag import ask_rag


st.set_page_config(
    page_title="FutureBrew RAG AI Support",
    page_icon="☕",
    layout="wide",
)

# -----------------------------
# Session State Init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# -----------------------------
# Header
# -----------------------------
st.title("☕ FutureBrew RAG AI Support")
st.caption(
    "A local RAG-based customer support assistant powered by Chroma, HuggingFace embeddings, and Ollama."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=5, value=3, step=1)
    debug_mode = st.toggle("Debug Mode", value=False)

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    st.write("Source type: Local TXT documents")
    st.write("Vector DB: Chroma")
    st.write("LLM: Ollama / DeepSeek-R1")

    st.markdown("---")
    st.subheader("💡 Example Questions")

    example_questions = [
        "退款政策是什么？",
        "是否支持更换牛奶类型？",
        "制作需要多长时间？",
        "会员优惠怎么使用？",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    st.subheader("🧠 Project Highlights")
    st.markdown(
        """
- Local RAG pipeline
- Evidence-grounded answers
- Chroma semantic retrieval
- Ollama local inference
- Debug context inspection
"""
    )

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("💬 Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_question = st.chat_input("请输入你的问题，例如：退款政策是什么？")

    # If clicked example question
    if not user_question and st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = ""

    if user_question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run RAG
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    result = ask_rag(user_question, top_k=top_k)

                    answer = result.get("answer", "未返回答案。")
                    sources = result.get("sources", [])
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    context = result.get("context", "")

                    st.markdown("### ✅ Answer")
                    st.write(answer)

                    if sources:
                        st.markdown("### 📌 Evidence Sources")
                        for src in sources:
                            st.markdown(f"- `{src}`")

                    if retrieved_chunks:
                        st.markdown("### 🔎 Retrieved Chunks")
                        for i, chunk in enumerate(retrieved_chunks, start=1):
                            source = chunk.get("source", "unknown")
                            content = chunk.get("content", "")
                            with st.expander(f"Chunk {i} | Source: {source}"):
                                st.write(content)

                    if debug_mode:
                        st.markdown("### 🧪 Debug Context")
                        st.text_area(
                            "Context sent to LLM",
                            value=context,
                            height=250,
                        )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"发生错误：{e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

with right_col:
    st.subheader("📊 Retrieval Panel")

    st.info(
        f"""
**Current Settings**
- Top-K: {top_k}
- Debug Mode: {"On" if debug_mode else "Off"}
"""
    )

    st.markdown("### 📝 Suggested Demo Questions")
    st.markdown(
        """
- 退款政策是什么？
- 是否支持燕麦奶 / 豆奶替换？
- 下单后多久开始制作？
- 有哪些促销活动？
"""
    )

    st.markdown("### 🎯 What This Demo Shows")
    st.markdown(
        """
1. Retrieval from local knowledge base  
2. Evidence-grounded answer generation  
3. Source file traceability  
4. Debug-friendly RAG workflow  
"""
    )