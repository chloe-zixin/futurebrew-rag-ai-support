from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1) 用同一个 embedding 模型（必须一致）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2) 加载你刚刚生成的向量库
vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# 3) 你要测试的问题（随便改）
query = "退款政策是什么？"

# 4) 检索最相关的3块内容
docs = vectorstore.similarity_search(query, k=3)

print("✅ 检索结果：")
for i, doc in enumerate(docs, start=1):
    print("\n====================")
    print(f"Top {i} | 来源: {doc.metadata.get('source')}")
    print(doc.page_content[:400])