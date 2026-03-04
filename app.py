import time
import requests
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===== 配置 =====
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "deepseek-r1:8b"

SYSTEM_PROMPT = """你是 FutureBrew AI咖啡店客服。
只基于【资料】回答，不要编造。
输出格式固定为 3 行以内要点：
- 制作前：
- 制作后：
- 异常情况/建议：
不要输出思考过程，不要寒暄。"""

# ===== 向量库加载（只会在首次运行时慢一点）=====
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vs = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    return vs

vectorstore = load_vectorstore()

def build_context(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        t = d.page_content.strip()
        text = (t[:250] + "\n...\n" + t[-250:]) if len(t) > 520 else t
        blocks.append(f"[{i}] 来源: {src}\n{text}")
    return "\n\n".join(blocks)

def ollama_chat(system_prompt: str, user_prompt: str, retries: int = 8) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.2,
            "num_ctx": 1536,
            "num_predict": 512,
        },
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            if r.status_code == 200:
                content = (r.json().get("message", {}) or {}).get("content", "").strip()
                if content:
                    return content
                # 空回答兜底再来一次
                payload["options"]["num_predict"] = 768
                r2 = requests.post(OLLAMA_URL, json=payload, timeout=180)
                if r2.status_code == 200:
                    return (r2.json().get("message", {}) or {}).get("content", "").strip()
                last_err = f"HTTP {r2.status_code}: {r2.text}"
            elif r.status_code == 503:
                time.sleep(1.2 * attempt)
                continue
            else:
                last_err = f"HTTP {r.status_code}: {r.text}"
        except Exception as e:
            last_err = str(e)
            time.sleep(1.2 * attempt)

    return f"（调用 Ollama 失败：{last_err}）"

def retrieve_docs(question: str):
    # 查询增强：退款/取消更偏向 FAQ
    if ("退款" in question) or ("取消" in question):
        query = question + " 退款 取消 政策 订单"
        allowed = {"faq.txt", "product_manual.txt"}
    else:
        query = question
        allowed = None

    raw_docs = vectorstore.similarity_search(query, k=8)

    if allowed:
        docs = [d for d in raw_docs if d.metadata.get("source") in allowed][:3]
        if not docs:
            docs = raw_docs[:3]
    else:
        docs = raw_docs[:3]

    return docs

# ===== UI =====
st.set_page_config(page_title="FutureBrew RAG 客服", page_icon="☕", layout="centered")
st.title("☕ FutureBrew RAG 客服（本地 Ollama + 向量检索）")
st.caption("输入问题 → 检索知识库 → 调用本地大模型生成客服回答，并展示引用证据。")

question = st.text_input("请输入你的问题（例如：退款政策是什么？）", value="退款政策是什么？")

col1, col2 = st.columns([1, 1])
with col1:
    topk = st.slider("检索证据条数", min_value=2, max_value=5, value=3)
with col2:
    show_context = st.checkbox("显示拼接后的上下文（调试）", value=False)

if st.button("提问 🚀"):
    if not question.strip():
        st.warning("先输入一个问题🙂")
    else:
        docs = retrieve_docs(question.strip())[:topk]
        context = build_context(docs)

        user_prompt = f"""【资料】
{context}

【问题】
{question}

【要求】
1) 只用资料回答，不要编造。
2) 若资料不足，必须明确说“资料中未找到明确说明”。
3) 输出固定为 3 行以内要点（制作前/制作后/异常情况或建议）。
请输出最终答复："""

        with st.spinner("正在检索 + 生成回答..."):
            answer = ollama_chat(SYSTEM_PROMPT, user_prompt)

        st.subheader("🤖 客服回答")
        st.write(answer if answer.strip() else "（模型返回空内容，建议再点一次提问或降低模型压力）")

        st.subheader("📌 引用证据")
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            st.markdown(f"**[{i}] 来源：`{src}`**")
            st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

        if show_context:
            st.subheader("🧩 拼接上下文（调试）")
            st.code(context)