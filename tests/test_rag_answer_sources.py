import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app import rag


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_filter = None

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        self.last_query = query
        self.last_filter = filter
        return self.results[:k]


class RagAnswerSourceTests(unittest.TestCase):
    def run_query(self, results, model_answer=None):
        store = FakeVectorStore(results)
        with patch("app.rag.load_vector_db", return_value=store), patch(
            "app.rag.ask_ollama", return_value=model_answer
        ) as ollama:
            result = rag.ask_rag("What is the Aurora-X17 warranty?")
        return result, ollama

    def render_app_result(self, result):
        app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
        with patch("app.rag.ask_rag", return_value=result):
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            app.chat_input[0].set_value("test question").run()
        self.assertFalse(app.exception, [exc.message for exc in app.exception])
        return app

    def test_supported_answer_keeps_its_actual_source_and_chunk(self):
        document = Document(
            page_content="Aurora-X17 is covered by a 17-month warranty.",
            metadata={"source": "aurora.pdf"},
        )
        result, ollama = self.run_query([(document, 0.8)], "The warranty is 17 months.")

        self.assertTrue(result["has_answer"])
        self.assertEqual(result["sources"], ["aurora.pdf"])
        self.assertEqual(
            result["retrieved_chunks"],
            [{"source": "aurora.pdf", "content": document.page_content}],
        )
        ollama.assert_called_once()

    def test_low_score_candidates_are_returned_only_as_no_answer_references(self):
        document = Document(
            page_content="A loosely related document excerpt.",
            metadata={"source": "reference.pdf"},
        )
        result, ollama = self.run_query(
            [(document, rag.MIN_RELEVANCE_SCORE - 0.01)]
        )

        self.assertFalse(result["has_answer"])
        self.assertEqual(result["answer"], rag.NO_ANSWER)
        self.assertEqual(result["sources"], ["reference.pdf"])
        self.assertEqual(
            result["retrieved_chunks"],
            [{"source": "reference.pdf", "content": document.page_content}],
        )
        self.assertEqual(result["context"], "")
        ollama.assert_not_called()

    def test_model_refusal_does_not_label_candidates_as_answer_evidence(self):
        document = Document(
            page_content="Candidate text that does not answer the question.",
            metadata={"source": "candidate.pdf"},
        )
        result, ollama = self.run_query(
            [(document, 0.8)],
            f"{rag.NO_ANSWER} The available text does not answer this.",
        )

        self.assertFalse(result["has_answer"])
        self.assertIn(rag.NO_ANSWER, result["answer"])
        self.assertEqual(result["sources"], ["candidate.pdf"])
        self.assertEqual(result["retrieved_chunks"][0]["source"], "candidate.pdf")
        ollama.assert_called_once()

    def test_empty_retrieval_returns_clean_no_answer_result(self):
        result, ollama = self.run_query([])

        self.assertFalse(result["has_answer"])
        self.assertEqual(result["answer"], rag.NO_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(result["context"], "")
        ollama.assert_not_called()

    def test_no_answer_candidates_are_labeled_as_references_in_ui(self):
        app = self.render_app_result(
            {
                "answer": rag.NO_ANSWER,
                "has_answer": False,
                "sources": ["candidate.pdf"],
                "retrieved_chunks": [
                    {"source": "candidate.pdf", "content": "Candidate excerpt."}
                ],
                "context": "",
            }
        )
        info_messages = [item.value for item in app.info]
        headings = [str(item.value) for item in app.markdown]
        expander_labels = [item.label for item in app.expander]

        self.assertIn(
            "知识库中没有找到足以支持该问题的证据",
            info_messages,
        )
        self.assertTrue(any("检索参考片段" in text for text in headings))
        self.assertIn("参考片段 1 | 来源: candidate.pdf", expander_labels)
        self.assertFalse(any("Evidence Sources" in text for text in headings))

    def test_empty_retrieval_ui_shows_no_candidate_message(self):
        app = self.render_app_result(
            {
                "answer": rag.NO_ANSWER,
                "has_answer": False,
                "sources": [],
                "retrieved_chunks": [],
                "context": "",
            }
        )
        info_messages = [item.value for item in app.info]

        self.assertIn(
            "知识库中没有找到足以支持该问题的证据",
            info_messages,
        )
        self.assertIn("未检索到可供参考的片段。", info_messages)

    def test_legacy_result_without_has_answer_preserves_refusal_semantics(self):
        app = self.render_app_result(
            {
                "answer": f"{rag.NO_ANSWER} 现有资料未说明具体条件。",
                "sources": ["candidate.pdf"],
                "retrieved_chunks": [
                    {"source": "candidate.pdf", "content": "Candidate excerpt."}
                ],
                "context": "",
            }
        )
        headings = [str(item.value) for item in app.markdown]

        self.assertTrue(
            any("知识库中没有找到足以支持该问题的证据" == item.value for item in app.info)
        )
        self.assertTrue(any("检索参考片段" in text for text in headings))
        self.assertFalse(any("Evidence Sources" in text for text in headings))

    def test_supported_answer_ui_keeps_real_source_label(self):
        app = self.render_app_result(
            {
                "answer": "Aurora-X17 is covered for 17 months.",
                "has_answer": True,
                "sources": ["aurora.pdf"],
                "retrieved_chunks": [
                    {"source": "aurora.pdf", "content": "17-month warranty."}
                ],
                "context": "[Source: aurora.pdf] 17-month warranty.",
            }
        )
        headings = [str(item.value) for item in app.markdown]
        expander_labels = [item.label for item in app.expander]

        self.assertTrue(any("Evidence Sources" in text for text in headings))
        self.assertTrue(any("aurora.pdf" in text for text in headings))
        self.assertIn("Chunk 1 | Source: aurora.pdf", expander_labels)


if __name__ == "__main__":
    unittest.main()
