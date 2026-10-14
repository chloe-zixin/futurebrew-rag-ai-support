from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.rag import load_vector_db


MAX_PDF_BYTES = 10 * 1024 * 1024
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 80


class PdfIngestionResult(TypedDict):
    status: Literal["added", "duplicate"]
    document_id: str
    filename: str
    chunk_count: int
    ids: list[str]


def _validate_pdf(file_bytes: bytes, filename: str) -> str:
    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise ValueError("上传文件为空，请选择非空 PDF 文件。")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF 文件超过 10 MB，请压缩后重新上传。")

    safe_filename = Path(filename.replace("\\", "/")).name.strip()
    if not safe_filename or Path(safe_filename).suffix.lower() != ".pdf":
        raise ValueError("仅支持 PDF 文件，请检查文件名和文件类型。")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("文件内容不是有效的 PDF，请重新选择文件。")
    return safe_filename


def extract_pdf_text(file_bytes: bytes, filename: str = "upload.pdf") -> str:
    _validate_pdf(file_bytes, filename)

    try:
        reader = PdfReader(BytesIO(file_bytes), strict=False)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF，请先解除密码保护后重新上传。")
        if not reader.pages:
            raise ValueError("PDF 不包含页面，无法提取文本。")
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "PDF 无法解析，文件可能已损坏或格式不受支持。"
        ) from exc

    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError(
            "PDF 中没有可提取的文本；扫描版 PDF 暂不支持 OCR，请上传可搜索文本的 PDF。"
        )
    return text


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    vectorstore: Chroma | None = None,
) -> PdfIngestionResult:
    safe_filename = _validate_pdf(file_bytes, filename)
    document_id = sha256(file_bytes).hexdigest()
    text = extract_pdf_text(file_bytes, safe_filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("PDF 文本无法切分为有效内容，请检查文件后重试。")
    store = vectorstore if vectorstore is not None else load_vector_db()
    expected_ids = [f"pdf-{document_id}-{index}" for index in range(len(chunks))]
    try:
        existing = store.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
    except Exception as exc:
        raise RuntimeError("无法检查 Chroma 知识库中的 PDF 记录，请检查数据库。") from exc

    existing_ids = set(existing.get("ids", []))
    if set(expected_ids).issubset(existing_ids):
        metadata = existing.get("metadatas", [])
        stored_filename = next(
            (
                item.get("source")
                for item in metadata
                if item and item.get("source")
            ),
            safe_filename,
        )
        return {
            "status": "duplicate",
            "document_id": document_id,
            "filename": stored_filename,
            "chunk_count": len(expected_ids),
            "ids": [],
        }

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": safe_filename,
                "document_id": document_id,
                "document_type": "uploaded_pdf",
                "chunk_index": index,
            },
        )
        for index, chunk in enumerate(chunks)
        if expected_ids[index] not in existing_ids
    ]
    new_ids = [
        expected_ids[index]
        for index in range(len(chunks))
        if expected_ids[index] not in existing_ids
    ]
    try:
        store.add_documents(documents=documents, ids=new_ids)
        stored = store.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
    except Exception as exc:
        raise RuntimeError(
            "PDF 已解析，但写入或校验 Chroma 知识库失败，请检查数据库状态。"
        ) from exc

    if not set(expected_ids).issubset(set(stored.get("ids", []))):
        raise RuntimeError("PDF 分块写入知识库后校验失败，请检查 Chroma 数据库。")

    return {
        "status": "added",
        "document_id": document_id,
        "filename": safe_filename,
        "chunk_count": len(expected_ids),
        "ids": new_ids,
    }


def delete_pdf_document(document_id: str) -> int:
    if len(document_id) != 64 or any(
        character not in "0123456789abcdef" for character in document_id
    ):
        raise ValueError("文档 ID 格式无效，未删除任何知识库记录。")

    store = load_vector_db()
    stored = store.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    ids = stored.get("ids", [])
    if ids:
        store.delete(ids=ids)
    return len(ids)
