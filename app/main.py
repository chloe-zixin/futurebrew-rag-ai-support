import time
import requests
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="FutureBrew RAG 客服", page_icon="☕")

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

# ===== Chat UI =====
st.title("☕ FutureBrew AI 客服")
st.caption("本地 Ollama + 向量检索的 RAG 智能客服 Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息（历史轮次只显示纯文本）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问点什么，比如：退款政策是什么？"):
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG 检索
    docs = retrieve_docs(prompt)
    context = build_context(docs)

    user_prompt = f"""【资料】
{context}

【问题】
{prompt}

【要求】
1) 只用资料回答，不要编造。
2) 若资料不足，必须说“资料中未找到明确说明”。
3) 输出固定为3行以内要点。
"""

    # 模型回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer = ollama_chat(SYSTEM_PROMPT, user_prompt)

        st.markdown(answer)

        with st.expander("📌 查看引用证据"):
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "unknown")
                snippet = d.page_content[:300]
                if len(d.page_content) > 300:
                    snippet += "..."
                st.markdown(f"**[{i}] 来源：{src}**")
                st.write(snippet)

    st.session_state.messages.append({"role": "assistant", "content": answer})