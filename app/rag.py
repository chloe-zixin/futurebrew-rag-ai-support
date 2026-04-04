from typing import Dict, List, Any

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
import requests


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
VECTOR_DB_DIR = "vector_db"
OLLAMA_MODEL = "deepseek-r1:8b"


def load_vector_db() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    return vectordb


def build_prompt(question: str, context: str) -> str:
    return f"""
你是 FutureBrew 咖啡店的智能客服助手。

请严格根据提供的知识库内容回答问题，不要编造信息。
如果知识库中没有明确答案，请直接说明“知识库中暂无明确信息”。

【用户问题】
{question}

【知识库上下文】
{context}

请用简洁、清晰、结构化的中文回答。
""".strip()


def ask_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return data["message"]["content"]

    except Exception as e:
        raise RuntimeError(f"Ollama HTTP 调用失败：{str(e)}")
        

def ask_rag(question: str, top_k: int = 3) -> Dict[str, Any]:
    vectordb = load_vector_db()

    docs = vectordb.similarity_search(question, k=top_k)

    retrieved_chunks: List[Dict[str, str]] = []
    sources = []

    context_parts = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")

        retrieved_chunks.append({
            "source": source,
            "content": content
        })

        sources.append(source)
        context_parts.append(f"[Source: {source}]\n{content}")

    # 去重但保持顺序
    dedup_sources = list(dict.fromkeys(sources))

    context = "\n\n".join(context_parts)
    prompt = build_prompt(question, context)
    answer = ask_ollama(prompt)

    return {
        "answer": answer,
        "sources": dedup_sources,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
    }