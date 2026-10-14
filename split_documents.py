import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "documents"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

all_chunks = []

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCS_PATH, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "chunk_id": i,
                "content": chunk
            })

print(f"总共生成 {len(all_chunks)} 个分块")

for chunk in all_chunks[:3]:
    print("\n====================")
    print(f"来源文件: {chunk['source']}")
    print(f"块编号: {chunk['chunk_id']}")
    print(chunk["content"][:200])