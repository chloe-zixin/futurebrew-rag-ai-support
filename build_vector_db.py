import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_PATH = "documents"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 变小一点，更精确
    chunk_overlap=80
)

documents = []

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCS_PATH, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # ✅ 对每个文件单独分块，并加上来源 metadata
        docs = text_splitter.create_documents(
            [text],
            metadatas=[{"source": filename}]
        )
        documents.extend(docs)

print(f"生成 {len(documents)} 个文档块")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# ✅ 重新建库：建议先删掉旧的 vector_db 文件夹
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="vector_db"
)

print("向量数据库创建完成！")