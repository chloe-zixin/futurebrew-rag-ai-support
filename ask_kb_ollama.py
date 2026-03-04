import time
import requests

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "deepseek-r1:8b"

SYSTEM_PROMPT = """你是 FutureBrew AI咖啡店客服。
只基于【资料】回答，不要编造。
输出格式固定为 3 行以内要点：
- 制作前：
- 制作后：
- 异常情况/建议：
最后必须加一行：引用：[编号]（编号来自资料块前的 [1][2][3]）
不要输出思考过程，不要寒暄。"""

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)

def build_context(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        t = d.page_content.strip()
        text = (t[:250] + "\n...\n" + t[-250:]) if len(t) > 520 else t
        blocks.append(f"[{i}] 来源: {src}\n{text}")
    return "\n\n".join(blocks)

def ollama_chat_request(system_prompt: str, user_prompt: str, retries: int = 8) -> str:
    def call_once(num_predict: int):
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
                "num_predict": num_predict,
            },
        }
        return requests.post(OLLAMA_URL, json=payload, timeout=180)

    last_status = None
    last_text = None

    for attempt in range(1, retries + 1):
        try:
            r = call_once(num_predict=512)
            last_status, last_text = r.status_code, r.text

            if r.status_code == 200:
                data = r.json()
                content = (data.get("message", {}) or {}).get("content", "").strip()

                if not content:
                    r2 = call_once(num_predict=768)
                    last_status, last_text = r2.status_code, r2.text
                    if r2.status_code == 200:
                        data2 = r2.json()
                        content2 = (data2.get("message", {}) or {}).get("content", "").strip()
                        return content2
                    if r2.status_code == 503:
                        time.sleep(1.5 * attempt)
                        continue
                    raise RuntimeError(f"Ollama HTTP {r2.status_code}: {r2.text}")

                return content

            if r.status_code == 503:
                time.sleep(1.5 * attempt)
                continue

            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text}")

        except requests.exceptions.RequestException as e:
            time.sleep(1.5 * attempt)
            if attempt == retries:
                raise RuntimeError(f"请求 Ollama 失败：{e}") from e

    raise RuntimeError(f"Ollama 多次重试仍失败。last_status={last_status}, last_text={last_text}")

def answer_question(question: str):
    # ✅ 查询增强：让退款类问题更容易把 FAQ 召回上来
    if ("退款" in question) or ("取消" in question):
        query = question + " 退款 取消 政策 订单"
    else:
        query = question

    raw_docs = vectorstore.similarity_search(query, k=8)

    if ("退款" in question) or ("取消" in question):
        allowed = {"faq.txt", "product_manual.txt"}
        docs = [d for d in raw_docs if d.metadata.get("source") in allowed][:3]
        if not docs:
            docs = raw_docs[:3]
    else:
        docs = raw_docs[:3]

    print("\n📌 检索到的资料片段：")
    for i, d in enumerate(docs, start=1):
        preview = d.page_content[:120].replace("\n", " ")
        print(f"- [{i}] {d.metadata.get('source')} | {preview}...")

    context = build_context(docs)

    user_prompt = f"""【资料】
{context}

【问题】
{question}

【要求】
1) 只用资料回答，不要编造。
2) 若资料不足，必须明确说“资料中未找到明确说明”。
3) 输出固定为 3 行以内要点（制作前/制作后/异常情况或建议）。
4) 末尾加一行：引用：[编号]（编号来自资料块前的 [1][2][3]）。
请输出最终答复："""

    answer = ollama_chat_request(SYSTEM_PROMPT, user_prompt)

    print("\n🤖 最终回答：")
    if answer.strip():
        print(answer)
    else:
        print("（模型这次返回了空内容；建议再问一次，或降低模型压力/换小模型）")

if __name__ == "__main__":
    while True:
        q = input("\n请输入你的问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        answer_question(q)