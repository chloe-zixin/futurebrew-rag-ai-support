import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from app import rag
from app.pdf_ingestion import (
    MAX_PDF_BYTES,
    delete_pdf_document,
    extract_pdf_text,
    ingest_pdf,
)


def create_text_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


class PdfUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = rag.load_vector_db()
        cls.baseline_ids = set(cls.store.get().get("ids", []))

    def test_extract_split_index_retrieve_and_preserve_existing_records(self):
        unique_fact = "Test product Aurora-X17 has a warranty period of 17 months."
        pdf_bytes = create_text_pdf(unique_fact)
        document_id = hashlib.sha256(pdf_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "aurora_test.pdf"
            pdf_path.write_bytes(pdf_bytes)
            extracted = extract_pdf_text(pdf_path.read_bytes(), pdf_path.name)
            self.assertIn(unique_fact, extracted)

            try:
                result = ingest_pdf(
                    pdf_path.read_bytes(),
                    pdf_path.name,
                    vectorstore=self.store,
                )
                self.assertEqual(result["status"], "added")
                self.assertGreater(result["chunk_count"], 0)
                self.assertEqual(result["filename"], "aurora_test.pdf")

                during_upload_ids = set(self.store.get().get("ids", []))
                self.assertTrue(self.baseline_ids.issubset(during_upload_ids))
                self.assertEqual(
                    len(during_upload_ids),
                    len(self.baseline_ids) + result["chunk_count"],
                )

                stored = self.store.get(
                    where={"document_id": document_id},
                    include=["documents", "metadatas"],
                )
                self.assertEqual(len(stored["ids"]), result["chunk_count"])
                self.assertEqual(
                    {metadata["source"] for metadata in stored["metadatas"]},
                    {"aurora_test.pdf"},
                )
                self.assertTrue(
                    all(
                        metadata["document_type"] == "uploaded_pdf"
                        for metadata in stored["metadatas"]
                    )
                )
                self.assertTrue(
                    any(unique_fact in document for document in stored["documents"])
                )

                duplicate = ingest_pdf(
                    pdf_path.read_bytes(),
                    "renamed_aurora_test.pdf",
                    vectorstore=self.store,
                )
                self.assertEqual(duplicate["status"], "duplicate")
                self.assertEqual(duplicate["ids"], [])
                self.assertEqual(
                    set(self.store.get().get("ids", [])), during_upload_ids
                )

                with patch("app.rag.ask_ollama", return_value="17 months"):
                    answer = rag.ask_rag(
                        "What is the warranty period for Aurora-X17?",
                        metadata_filter={"document_id": document_id},
                    )
                self.assertEqual(answer["answer"], "17 months")
                self.assertEqual(answer["sources"], ["aurora_test.pdf"])
                self.assertTrue(
                    any(
                        unique_fact in chunk["content"]
                        for chunk in answer["retrieved_chunks"]
                    )
                )

                removed = delete_pdf_document(document_id)
                self.assertEqual(removed, result["chunk_count"])
                self.assertFalse(
                    self.store.get(where={"document_id": document_id})["ids"]
                )
            finally:
                # Delete only records bearing this test PDF's unique content hash.
                created = self.store.get(
                    where={"document_id": document_id}
                ).get("ids", [])
                if created:
                    self.store.delete(ids=created)

            self.assertEqual(set(self.store.get().get("ids", [])), self.baseline_ids)

    def test_rejects_empty_unsupported_and_corrupt_files(self):
        cases = [
            (b"", "empty.pdf"),
            (b"not a pdf", "wrong.txt"),
            (b"%PDF-1.4\nnot a valid document", "broken.pdf"),
            (b"%PDF-" + b"0" * MAX_PDF_BYTES, "oversized.pdf"),
        ]
        for file_bytes, filename in cases:
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError) as context:
                    ingest_pdf(file_bytes, filename)
                self.assertTrue(str(context.exception))

    def test_rejects_pdf_without_extractable_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with tempfile.SpooledTemporaryFile() as blank_pdf:
            writer.write(blank_pdf)
            blank_pdf.seek(0)
            with self.assertRaises(ValueError) as context:
                ingest_pdf(blank_pdf.read(), "blank.pdf")
            self.assertTrue(str(context.exception))

    def test_no_answer_still_skips_model_generation(self):
        with patch("app.rag.ask_ollama") as llm:
            result = rag.ask_rag("门店是否提供免费停车？")
        self.assertEqual(result["answer"], rag.NO_ANSWER)
        self.assertFalse(result["has_answer"])
        llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
