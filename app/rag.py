import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests

# Load environment variables from .env file if it exists
load_dotenv()


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
VECTOR_DB_DIR = "vector_db"
OLLAMA_MODEL = "deepseek-r1:8b"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_MODEL = "Qwen/Qwen3.5-4B"
MIN_RELEVANCE_SCORE = 0.2
NO_ANSWER = "知识库中暂无明确信息。"
ModelProvider = Literal["ollama", "deepseek", "siliconflow"]


@lru_cache(maxsize=1)
def load_vector_db() -> Chroma:
    vector_db_path = Path(VECTOR_DB_DIR)
    if not vector_db_path.is_dir():
        raise RuntimeError(
            f"知识库目录不存在：{vector_db_path.resolve()}。"
            "请在项目根目录运行 `python build_vector_db.py` 首次构建知识库"
            "（Windows 可用 `.\\venv\\Scripts\\python.exe build_vector_db.py`）。"
        )

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"无法加载嵌入模型 {EMBEDDING_MODEL}。"
            "首次运行需要连接 Hugging Face 下载模型；请检查网络后重试。"
            f"原始错误：{e}"
        ) from e

    try:
        vectordb = Chroma(
            persist_directory=str(vector_db_path),
            embedding_function=embeddings,
        )
        stored_ids = vectordb.get(limit=1).get("ids", [])
    except Exception as e:
        raise RuntimeError(
            f"无法打开或读取 Chroma 知识库：{vector_db_path.resolve()}。"
            "请确认目录中的数据库文件完整且可读。"
            f"原始错误：{e}"
        ) from e

    if not stored_ids:
        raise RuntimeError(
            f"Chroma 知识库为空：{vector_db_path.resolve()}。"
            "请在项目根目录运行 `python build_vector_db.py` 首次构建知识库"
            "（Windows 可用 `.\\venv\\Scripts\\python.exe build_vector_db.py`）。"
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


def ask_deepseek(prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置 DeepSeek API Key。请在环境变量中设置 DEEPSEEK_API_KEY，"
            "或在 Streamlit Secrets 中配置。"
        )

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
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

        if "choices" not in data or not data["choices"]:
            raise RuntimeError("DeepSeek API 返回了无效的响应格式")

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError("DeepSeek API Key 无效或已过期") from e
        elif e.response.status_code == 429:
            raise RuntimeError("DeepSeek API 请求频率超限，请稍后重试") from e
        else:
            raise RuntimeError(f"DeepSeek API HTTP 错误：{e.response.status_code}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("DeepSeek API 请求超时，请检查网络连接") from e
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 调用失败：{str(e)}") from e


def ask_siliconflow(prompt: str) -> str:
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置 SiliconFlow API Key。请在环境变量中设置 SILICONFLOW_API_KEY，"
            "或在 Streamlit Secrets 中配置。"
        )

    try:
        response = requests.post(
            SILICONFLOW_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": SILICONFLOW_MODEL,
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

        if "choices" not in data or not data["choices"]:
            raise RuntimeError("SiliconFlow API 返回了无效的响应格式")

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise RuntimeError("SiliconFlow API Key 无效或已过期") from e
        elif e.response.status_code == 402:
            raise RuntimeError("SiliconFlow API 余额不足或配额已用完") from e
        elif e.response.status_code == 429:
            raise RuntimeError("SiliconFlow API 请求频率超限，请稍后重试") from e
        else:
            raise RuntimeError(f"SiliconFlow API HTTP 错误：{e.response.status_code}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("SiliconFlow API 请求超时，请检查网络连接") from e
    except Exception as e:
        raise RuntimeError(f"SiliconFlow API 调用失败：{str(e)}") from e


def ask_rag(
    question: str,
    top_k: int = 3,
    metadata_filter: Dict[str, str] | None = None,
    model_provider: ModelProvider = "ollama",
) -> Dict[str, Any]:
    vectordb = load_vector_db()

    query = question
    allowed_sources = None
    search_k = top_k
    if "退款" in question or "取消" in question:
        query = f"{question} 退款 取消 政策 订单"
        if metadata_filter is None:
            allowed_sources = {"faq.txt", "product_manual.txt"}
        search_k = max(top_k, 8)

    results = vectordb.similarity_search_with_relevance_scores(
        query,
        k=search_k,
        filter=metadata_filter,
    )
    if allowed_sources is not None:
        results = [
            (doc, score)
            for doc, score in results
            if doc.metadata.get("source") in allowed_sources
        ]

    docs = [
        doc
        for doc, score in results
        if score >= MIN_RELEVANCE_SCORE
    ][:top_k]

    def format_chunks(documents: List[Any]) -> List[Dict[str, str]]:
        return [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
            }
            for doc in documents
        ]

    if not docs:
        candidate_chunks = format_chunks([doc for doc, _ in results[:top_k]])
        return {
            "answer": NO_ANSWER,
            "has_answer": False,
            "sources": list(
                dict.fromkeys(chunk["source"] for chunk in candidate_chunks)
            ),
            "retrieved_chunks": candidate_chunks,
            "context": "",
            "model_provider": model_provider,
        }

    retrieved_chunks = format_chunks(docs)
    sources = []
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk["source"]
        sources.append(source)
        context_parts.append(f"[Source: {source}]\n{chunk['content']}")

    # 去重但保持顺序
    dedup_sources = list(dict.fromkeys(sources))

    context = "\n\n".join(context_parts)
    prompt = build_prompt(question, context)

    # 根据模型提供方选择调用方式
    if model_provider == "deepseek":
        answer = ask_deepseek(prompt).strip()
    elif model_provider == "siliconflow":
        answer = ask_siliconflow(prompt).strip()
    else:  # 默认使用 ollama
        answer = ask_ollama(prompt).strip()

    has_answer = bool(answer) and NO_ANSWER not in answer
    if not answer:
        answer = NO_ANSWER

    return {
        "answer": answer,
        "has_answer": has_answer,
        "sources": dedup_sources,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "model_provider": model_provider,
    }