from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1) Embedding（必须和建库一致）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2) 加载向量库
vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)


def answer_question(query: str):
    # 3) 先多检索一些，再过滤（防止漏召回）
    raw_docs = vectorstore.similarity_search(query, k=8)

    # 4) 根据问题类型过滤来源
    if ("退款" in query) or ("取消" in query):
        allowed = {"faq.txt", "product_manual.txt", "training_guide.txt"}
        docs = [d for d in raw_docs if d.metadata.get("source") in allowed][:3]
        # 兜底：如果过滤后为空，就退回原始结果
        if not docs:
            docs = raw_docs[:3]
    else:
        docs = raw_docs[:3]

    # 5) 输出证据（引用来源）
    print("\n📌 依据（检索到的资料片段）:")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        excerpt = doc.page_content.strip().replace("\n", " ")
        print(f"\n- 证据{i}（来自 {source}）: {excerpt[:220]}...")

    # 6) 输出“客服回答”（不使用LLM，靠规则提炼）
    print("\n🤖 客服回答：")

    # ——退款/取消类：用模板生成更像客服的话术——
    if ("退款" in query) or ("取消" in query):
        # 从 docs 里提炼包含关键词的句子，避免太“内部流程”句子抢戏
        key_sentences = []
        for d in docs:
            src = d.metadata.get("source", "")
            text = d.page_content.replace("\n", " ").strip()

            for sent in text.split("。"):
                s = sent.strip()
                if not s:
                    continue

                if ("退款" in s) or ("取消" in s):
                    # 过滤一些特别“内部流程”的句子（可继续加规则）
                    if ("记录投诉" in s) or ("上报" in s):
                        continue
                    key_sentences.append((src, s + "。"))

        # 去重 + 按来源优先级排序：FAQ/手册优先
        priority = {"faq.txt": 0, "product_manual.txt": 1, "training_guide.txt": 2}

        uniq = []
        seen = set()
        for src, s in sorted(key_sentences, key=lambda x: priority.get(x[0], 99)):
            if s not in seen:
                seen.add(s)
                uniq.append((src, s))

        # 生成更像客服的回答（模板）
        before_make = ""   # 制作前取消/退款
        after_make = ""    # 制作完成后不支持退款
        exception = ""     # 异常情况补做/退款

        for src, s in uniq:
            if (("制作前" in s) or ("制作开始前" in s) or ("取消订单" in s)) and not before_make:
                before_make = s
            if (("制作完成" in s) or ("完成后" in s)) and not after_make:
                after_make = s
            if (("补做" in s) or ("异常" in s) or ("流程" in s)) and not exception:
                exception = s

        print("退款规则如下：")
        if before_make:
            print(f"- {before_make}")
        else:
            print("- 饮品制作开始前可取消订单并退款（以订单状态为准）。")

        if after_make:
            print(f"- {after_make}")
        else:
            print("- 饮品制作完成后原则上不支持退款。")

        if exception:
            print("- 如遇设备异常或出品问题，可联系门店按标准流程处理（补做/退款）。")

        print("建议：如需退款请尽量在制作开始前操作取消；如遇异常可联系门店工作人员处理。")
        return

    # ——非退款类：先给“可解释”的回答方式（不依赖LLM）——
    # 你也可以后面升级成 LLM 总结
    print("我已从知识库中检索到相关资料片段（见上方依据）。")
    print("如果你希望我整理成更精确的答复，请把问题问得更具体一点，例如：")
    print("- 甜度怎么选？")
    print("- 支付方式有哪些？")
    print("- 制作需要多久？")
    print("- 设备有哪些使用限制？")


if __name__ == "__main__":
    while True:
        q = input("\n请输入你的问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        answer_question(q)