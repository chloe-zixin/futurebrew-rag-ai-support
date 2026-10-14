import streamlit as st
from app.pdf_ingestion import ingest_pdf
from app.rag import NO_ANSWER, ask_rag, ModelProvider


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

if "active_uploaded_pdf" not in st.session_state:
    st.session_state.active_uploaded_pdf = None

if "model_provider" not in st.session_state:
    st.session_state.model_provider = "ollama"

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
    st.subheader("🤖 Model Provider")
    model_provider = st.radio(
        "Select LLM Provider",
        options=["ollama", "deepseek", "siliconflow"],
        format_func=lambda x: {
            "ollama": "Local Ollama",
            "deepseek": "DeepSeek Cloud API",
            "siliconflow": "SiliconFlow API (Qwen3.5-4B)"
        }.get(x, x),
        index=0 if st.session_state.model_provider == "ollama" else
              1 if st.session_state.model_provider == "deepseek" else 2,
        key="model_provider_radio"
    )
    st.session_state.model_provider = model_provider

    if model_provider == "deepseek":
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            st.warning(
                "⚠️ 未配置 DeepSeek API Key。请在环境变量中设置 DEEPSEEK_API_KEY，"
                "或在 Streamlit Secrets 中配置。"
            )
        else:
            st.success("✅ DeepSeek API Key 已配置")
    elif model_provider == "siliconflow":
        import os
        if not os.environ.get("SILICONFLOW_API_KEY"):
            st.warning(
                "⚠️ 未配置 SiliconFlow API Key。请在环境变量中设置 SILICONFLOW_API_KEY，"
                "或在 Streamlit Secrets 中配置。"
            )
        else:
            st.success("✅ SiliconFlow API Key 已配置")
    else:
        st.info("📡 使用本地 Ollama 模型，确保 Ollama 服务正在运行")

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    st.write("Source type: Local TXT documents and uploaded PDFs")
    st.write("Vector DB: Chroma")
    st.write("LLM: Ollama / DeepSeek-R1")

    with st.expander("📄 Upload a PDF to the knowledge base"):
        uploaded_pdf = st.file_uploader(
            "Choose a text-based PDF (maximum 10 MB)",
            type=["pdf"],
            key="knowledge_pdf_uploader",
        )
        process_pdf = st.button(
            "Add PDF to knowledge base",
            disabled=uploaded_pdf is None,
            use_container_width=True,
        )
        if process_pdf and uploaded_pdf is not None:
            try:
                ingestion = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name,
                )
                st.session_state.active_uploaded_pdf = {
                    "document_id": ingestion["document_id"],
                    "filename": ingestion["filename"],
                    "chunk_count": ingestion["chunk_count"],
                }
                if ingestion["status"] == "duplicate":
                    st.info(
                        f"`{ingestion['filename']}` is already in the knowledge "
                        f"base ({ingestion['chunk_count']} chunks); no duplicate "
                        "records were added."
                    )
                else:
                    st.success(
                        f"`{ingestion['filename']}` added successfully "
                        f"({ingestion['chunk_count']} text chunks)."
                    )
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(f"Could not add the PDF to the knowledge base: {exc}")

        active_pdf = st.session_state.active_uploaded_pdf
        if active_pdf:
            st.caption(
                f"Current PDF: {active_pdf['filename']} "
                f"({active_pdf['chunk_count']} chunks)"
            )
            search_uploaded_pdf = st.checkbox(
                "Limit questions to this PDF",
                value=True,
                key="search_uploaded_pdf_only",
            )
        else:
            search_uploaded_pdf = False

    st.markdown("---")
    st.subheader("💡 Example Questions")

    example_questions = [
        "退款政策是什么？",
        "是否支持更换牛奶类型？",
        "制作需要多长时间？",
        "会员优惠怎么使用？",
        "门店是否提供免费停车？",
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
                    metadata_filter = None
                    if search_uploaded_pdf and active_pdf:
                        metadata_filter = {
                            "document_id": active_pdf["document_id"]
                        }
                    result = ask_rag(
                        user_question,
                        top_k=top_k,
                        metadata_filter=metadata_filter,
                        model_provider=st.session_state.model_provider,
                    )

                    answer = result.get("answer", "未返回答案。")
                    has_answer = result.get(
                        "has_answer",
                        bool(answer) and NO_ANSWER not in answer,
                    )
                    sources = result.get("sources", [])
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    context = result.get("context", "")

                    if has_answer:
                        st.markdown("### ✅ Answer")
                        st.write(answer)
                        st.markdown("### 📌 Evidence Sources")
                        for src in sources:
                            st.markdown(f"- `{src}`")
                        if retrieved_chunks:
                            st.markdown("### 🔎 Retrieved Chunks")
                    else:
                        st.info("知识库中没有找到足以支持该问题的证据")
                        if retrieved_chunks:
                            st.markdown("### 🔎 检索参考片段")
                        else:
                            st.info("未检索到可供参考的片段。")

                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks, start=1):
                            source = chunk.get("source", "unknown")
                            content = chunk.get("content", "")
                            label = (
                                f"Chunk {i} | Source: {source}"
                                if has_answer
                                else f"参考片段 {i} | 来源: {source}"
                            )
                            with st.expander(label):
                                st.write(content)

                    if debug_mode and has_answer:
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

    model_display = {
        "ollama": "Local Ollama",
        "deepseek": "DeepSeek Cloud API",
        "siliconflow": "SiliconFlow API (Qwen3.5-4B)"
    }.get(st.session_state.model_provider, "Unknown")
    st.info(
        f"""
**Current Settings**
- Model: {model_display}
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
- 无答案验证：门店是否提供免费停车？
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