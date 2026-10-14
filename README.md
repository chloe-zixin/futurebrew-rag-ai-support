# ☕ FutureBrew RAG AI Support

A RAG-based customer support assistant with support for both local (Ollama) and cloud (DeepSeek API) LLM inference.

---

## 🧠 System Architecture

![Architecture](./assets/architecture.png)

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for AI customer support scenarios.

It retrieves relevant knowledge from a local document database and generates structured, evidence-grounded answers using **local LLM (Ollama) or cloud LLMs (DeepSeek API or SiliconFlow API)**.

---

## ✨ Features

* 📚 Local knowledge base (TXT documents)
* 🔍 Semantic retrieval with Chroma vector database
* 🧠 HuggingFace embeddings (bge-small-zh)
* 🤖 **Multi LLM support**: Local Ollama (DeepSeek-R1), Cloud DeepSeek API, or Cloud SiliconFlow API (Qwen3.5-4B)
* 🌐 Streamlit web UI (chat-style interface)
* 📌 Retrieved source files and evidence chunks
* 🎛 Adjustable retrieval count
* 🧪 Debug mode (view full context)
* 📄 Upload searchable PDFs and query their contents
* 🔐 Secure API key management via environment variables
* 🔄 Model switching between local and cloud providers

---

## 🧱 Project Structure

```text
rag_project/
├── app/
│   ├── __init__.py
│   ├── main.py          # Streamlit chat UI with model switching
│   ├── rag.py           # RAG pipeline logic (local + cloud LLM)
│   └── pdf_ingestion.py # PDF extraction, chunking, and Chroma append
├── documents/           # Knowledge base files
├── vector_db/           # Chroma vector DB (ignored by Git)
├── tests/
│   ├── test_pdf_upload.py
│   ├── test_rag_answer_sources.py
│   └── test_cloud_api.py # Cloud API mock tests
├── .streamlit/
│   └── config.toml      # Streamlit upload-size limit
├── .env.example         # Environment variable template
├── build_vector_db.py   # Build embeddings
├── split_documents.py   # Text splitting
├── test_retrieval.py    # Retrieval testing
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Load local text documents
2. Split into chunks
3. Convert to embeddings using BGE model
4. Store in Chroma vector DB
5. Retrieve top-k relevant chunks
6. Build context for LLM
7. Send prompt to LLM (Ollama local or DeepSeek cloud)
8. Generate structured answer
9. Display answer + evidence in UI

---

## 🖥️ Run the Project (Windows PowerShell)

Use Python 3.11.x. The existing project environment was created with Python 3.11.9.
Run these commands from the repository root (the directory containing this README).

### 1. Create an environment and install dependencies

```powershell
py -3.11 --version
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file lists the project's direct Python runtime dependencies with
the versions currently used by this repository.

### 2. Choose LLM Provider

#### Option A: Local Ollama (Default)

Install Ollama for Windows from [ollama.com/download/windows](https://ollama.com/download/windows),
then open a new PowerShell window and download the model:

```powershell
ollama pull deepseek-r1:8b
ollama list
```

Keep the Ollama application/service running while using the app. The current
Python code sends requests to `http://127.0.0.1:11434`.

#### Option B: Cloud DeepSeek API

**Privacy Note**: When using cloud mode, your questions and retrieved document fragments are sent to DeepSeek's API servers.

1. Get an API key from [DeepSeek Platform](https://platform.deepseek.com/)
2. Configure the API key using one of these methods:

**Method 1: Environment Variable (Recommended)**
```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"

# Or create a .env file in the project root:
# Copy .env.example to .env and fill in your key
```

**Method 2: Streamlit Secrets (For Deployment)**
- Go to Streamlit Cloud dashboard
- Navigate to your app settings
- Add `DEEPSEEK_API_KEY` in the Secrets section

#### Option C: Cloud SiliconFlow API

**Privacy Note**: When using cloud mode, your questions and retrieved document fragments are sent to SiliconFlow's API servers.

1. Get an API key from [SiliconFlow Platform](https://siliconflow.cn/)
2. Configure the API key using one of these methods:

**Method 1: Environment Variable (Recommended)**
```powershell
# Windows PowerShell
$env:SILICONFLOW_API_KEY="your_api_key_here"

# Or create a .env file in the project root:
# Copy .env.example to .env and fill in your key
```

**Method 2: Streamlit Secrets (For Deployment)**
- Go to Streamlit Cloud dashboard
- Navigate to your app settings
- Add `SILICONFLOW_API_KEY` in the Secrets section

### 3. Build the local knowledge base and prepare embeddings

```powershell
.\venv\Scripts\python.exe build_vector_db.py
```

This reads the TXT files in `documents/`, creates chunks, and writes the Chroma
database to `vector_db/`. The first build also loads
`BAAI/bge-small-zh-v1.5`; `HuggingFaceEmbeddings` downloads it from Hugging Face
when it is not already cached. Internet access to Hugging Face is therefore
needed on the first build. The vector database is ignored by Git and is created
locally.

Run this build command for a fresh checkout with no `vector_db/` directory. The
current builder is not an update-safe rebuild command; do not rerun it against
an existing knowledge base until its update behavior has been addressed.

### 4. Start Streamlit

```powershell
.\venv\Scripts\python.exe -m streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501).

The Streamlit app expects the local Chroma database and the cached embedding model
to be available. If the database is missing or empty, the app displays a message
directing you to the first-build command.

### Switching Between Models

In the sidebar under **Model Provider**, select:
- **Local Ollama**: Uses the local Ollama service (requires Ollama running)
- **DeepSeek Cloud API**: Uses DeepSeek's cloud API (requires API key configured)
- **SiliconFlow API (Qwen3.5-4B)**: Uses SiliconFlow's cloud API (requires API key configured)

The app will show configuration warnings if the selected provider is not properly set up.

---

## 📄 Upload and Query PDFs

1. Open **Upload a PDF to the knowledge base** in the sidebar and select a
   text-based PDF no larger than 10 MB.
2. Select **Add PDF to knowledge base**. The app extracts its text, splits it
   using the existing 500-character chunk size and 80-character overlap,
   embeds it with the existing BGE model, and appends it to Chroma.
3. Ask a question in the chat. **Limit questions to this PDF** is enabled by
   default after a successful upload; turn it off to search the full knowledge
   base.

An identical PDF (same file bytes) is detected by SHA-256 and is not indexed a
second time. A different PDF with the same filename is stored separately.
Uploaded chunks retain the filename, content hash (`document_id`), chunk index,
and `uploaded_pdf` document type in Chroma metadata, enabling safe targeted
replacement or deletion later; the UI does not currently provide document
management. Uploading never rebuilds or clears the existing database.

Only PDFs with extractable text are supported. Scanned/image-only PDFs require
OCR and are rejected with a clear message. Empty, malformed, encrypted, and
over-10-MB files are rejected. The first upload may take time while BGE loads;
the selected LLM (local or cloud) must be available for answer generation.

---

## 💬 Example Questions

* 退款政策是什么？
* 是否支持更换牛奶类型？
* 制作需要多长时间？
* 无答案验证：门店是否提供免费停车？

---

## 🧾 Example Output

The system returns a concise answer, the retrieved source filenames, and the
retrieved text chunks. Queries below the retrieval relevance threshold return
"知识库中暂无明确信息" without calling the chat model. The current prompt
requests a structured Chinese answer; it does not enforce a fixed format
or validate citations.

实际多来源运行示例（问题：燕麦奶替换和设备日常清洁维护）：

> 燕麦奶是支持的牛奶类型之一，用户可在下单页面选择；具体供应以门店为准。
>
> 设备维护包括每日营业结束后运行深度清洁、清理咖啡渣盒和废水盒、擦拭外部表面、每周检查研磨刀盘，以及每月由专业人员维护。

本次回答检索并展示了 `product_manual.txt`、`faq.txt` 和
`training_guide.txt`。模型生成措辞可能因运行而异；知识库无充分依据时，
界面显示"知识库中暂无明确信息"并跳过模型生成。

---

## ⚠️ Notes

* `venv/` and `vector_db/` are excluded from Git.
* For local Ollama mode: Ollama must be running and have `deepseek-r1:8b` available.
* For cloud DeepSeek mode: API key must be configured; questions and context are sent to DeepSeek servers.
* The first embedding-model download and database build may take time.
* PDF upload tests use the existing local Chroma database and remove only the
  uniquely identified test document when complete.

---

## 🧪 Testing

### Run all tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test coverage

* **test_pdf_upload.py**: PDF upload, duplicate detection, metadata, and cleanup
* **test_rag_answer_sources.py**: RAG answer validation, source tracing, UI rendering
* **test_cloud_api.py**: Cloud API functionality (DeepSeek and SiliconFlow, uses mocks, no real API calls)

The PDF upload suite generates temporary searchable, blank, corrupt, and
oversized PDF inputs. It uses the real local BGE embeddings and Chroma
collection to check ingestion, duplicate detection, metadata, filtered RAG
retrieval, and targeted cleanup. The cloud API tests use mocks to avoid
actual API calls and charges.

---

## 🔒 Security & Privacy

* API keys are never committed to Git (see `.gitignore`)
* Use environment variables or Streamlit Secrets for sensitive configuration
* Cloud mode sends data to external API servers (DeepSeek)
* Local mode keeps all data on your machine
* Uploaded PDFs are stored locally in the vector database

---

## 🔮 Future Improvements

* Streaming response output
* Collapsible evidence UI
* FastAPI backend refactor
* Multi-user support
* More cloud LLM providers

---

## 📌 Tech Stack

* Python 3.11+
* Streamlit 1.55.0
* LangChain (Community, HuggingFace, Text Splitters)
* Chroma 1.5.2
* HuggingFace Embeddings (BAAI/bge-small-zh-v1.5)
* Ollama (Local)
* DeepSeek API (Cloud)
* SiliconFlow API (Cloud)
* python-dotenv (Environment configuration)

---

## 👤 Author

Catherine
AI Engineering / System-Oriented Developer
