# -*- coding: utf-8 -*-
import unittest
import requests
from unittest.mock import patch, Mock
from app import rag


class CloudApiTests(unittest.TestCase):
    """Test cloud API (DeepSeek and SiliconFlow) functionality with mocks to avoid real API calls."""

    def test_deepseek_with_valid_api_key(self):
        """Test DeepSeek API call with valid key (mocked)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test answer about milk options"
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                result = rag.ask_deepseek("是否支持更换牛奶类型？")

                mock_post.assert_called_once()
                self.assertIn("milk", result)
                self.assertTrue(len(result) > 0)

    def test_deepseek_without_api_key(self):
        """Test DeepSeek API call without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError) as context:
                rag.ask_deepseek("测试问题")

            self.assertIn("未配置 DeepSeek API Key", str(context.exception))

    def test_deepseek_invalid_api_key(self):
        """Test DeepSeek API call with invalid key (401 error)."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'invalid_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_deepseek("测试问题")

                # The error message should indicate authentication failure
                self.assertTrue("401" in str(context.exception) or "Unauthorized" in str(context.exception))

    def test_deepseek_rate_limit(self):
        """Test DeepSeek API rate limit (429 error)."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("429 Too Many Requests")

        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_deepseek("测试问题")

                # The error message should indicate rate limiting
                self.assertTrue("429" in str(context.exception) or "Too Many" in str(context.exception))

    def test_deepseek_timeout(self):
        """Test DeepSeek API timeout."""
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', side_effect=requests.exceptions.Timeout("Timeout")):
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_deepseek("测试问题")

                self.assertIn("超时", str(context.exception))

    def test_rag_with_deepseek_provider(self):
        """Test RAG pipeline with DeepSeek provider (mocked)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "测试答案"
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        # Mock vector store and embedding
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="测试内容", metadata={"source": "test.txt"}), 0.8)
        ]
        mock_vectorstore.get.return_value = {"ids": ["1"]}

        with patch('app.rag.load_vector_db', return_value=mock_vectorstore):
            with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
                with patch('app.rag.requests.post', return_value=mock_response):
                    result = rag.ask_rag("测试问题", model_provider="deepseek")

                    self.assertEqual(result["model_provider"], "deepseek")
                    self.assertTrue(result["has_answer"])

    def test_rag_without_answer_with_deepseek(self):
        """Test RAG with no relevant documents skips model call even with DeepSeek."""
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_vectorstore = MagicMock()
        # Return low relevance scores
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="无关内容", metadata={"source": "test.txt"}), 0.1)
        ]
        mock_vectorstore.get.return_value = {"ids": ["1"]}

        with patch('app.rag.load_vector_db', return_value=mock_vectorstore):
            with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
                with patch('app.rag.requests.post') as mock_post:
                    result = rag.ask_rag("测试问题", model_provider="deepseek")

                    # Should not call the API when no relevant docs
                    mock_post.assert_not_called()
                    self.assertFalse(result["has_answer"])
                    self.assertEqual(result["answer"], rag.NO_ANSWER)

    # --- SiliconFlow Tests ---

    def test_siliconflow_with_valid_api_key(self):
        """Test SiliconFlow API call with valid key (mocked)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test answer about Qwen"
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                result = rag.ask_siliconflow("测试问题")

                mock_post.assert_called_once()
                self.assertIn("Qwen", result)
                self.assertTrue(len(result) > 0)

    def test_siliconflow_without_api_key(self):
        """Test SiliconFlow API call without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError) as context:
                rag.ask_siliconflow("测试问题")

            self.assertIn("未配置 SiliconFlow API Key", str(context.exception))

    def test_siliconflow_invalid_api_key(self):
        """Test SiliconFlow API call with invalid key (401 error)."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'invalid_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_siliconflow("测试问题")

                self.assertTrue("401" in str(context.exception) or "Unauthorized" in str(context.exception))

    def test_siliconflow_insufficient_balance(self):
        """Test SiliconFlow API insufficient balance (402 error)."""
        mock_response = Mock()
        mock_response.status_code = 402
        mock_response.raise_for_status.side_effect = Exception("402 Payment Required")

        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_siliconflow("测试问题")

                self.assertTrue("402" in str(context.exception) or "余额不足" in str(context.exception))

    def test_siliconflow_rate_limit(self):
        """Test SiliconFlow API rate limit (429 error)."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("429 Too Many Requests")

        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response) as mock_post:
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_siliconflow("测试问题")

                self.assertTrue("429" in str(context.exception) or "Too Many" in str(context.exception))

    def test_siliconflow_timeout(self):
        """Test SiliconFlow API timeout."""
        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', side_effect=requests.exceptions.Timeout("Timeout")):
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_siliconflow("测试问题")

                self.assertIn("超时", str(context.exception))

    def test_siliconflow_invalid_response(self):
        """Test SiliconFlow API returns invalid response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = Mock()

        with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
            with patch('app.rag.requests.post', return_value=mock_response):
                with self.assertRaises(RuntimeError) as context:
                    rag.ask_siliconflow("测试问题")

                self.assertIn("无效的响应格式", str(context.exception))

    def test_rag_with_siliconflow_provider(self):
        """Test RAG pipeline with SiliconFlow provider (mocked)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "测试答案"
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="测试内容", metadata={"source": "test.txt"}), 0.8)
        ]
        mock_vectorstore.get.return_value = {"ids": ["1"]}

        with patch('app.rag.load_vector_db', return_value=mock_vectorstore):
            with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
                with patch('app.rag.requests.post', return_value=mock_response):
                    result = rag.ask_rag("测试问题", model_provider="siliconflow")

                    self.assertEqual(result["model_provider"], "siliconflow")
                    self.assertTrue(result["has_answer"])

    def test_rag_without_answer_with_siliconflow(self):
        """Test RAG with no relevant documents skips model call even with SiliconFlow."""
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="无关内容", metadata={"source": "test.txt"}), 0.1)
        ]
        mock_vectorstore.get.return_value = {"ids": ["1"]}

        with patch('app.rag.load_vector_db', return_value=mock_vectorstore):
            with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
                with patch('app.rag.requests.post') as mock_post:
                    result = rag.ask_rag("测试问题", model_provider="siliconflow")

                    mock_post.assert_not_called()
                    self.assertFalse(result["has_answer"])
                    self.assertEqual(result["answer"], rag.NO_ANSWER)

    def test_siliconflow_wifi_frequency_question(self):
        """Test SiliconFlow with a specific Wi-Fi frequency question (regression test)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Aurora-X7 支持 2.4GHz 和 5GHz 双频 Wi-Fi"
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="Aurora-X7 支持 2.4GHz 和 5GHz 双频 Wi-Fi", metadata={"source": "aurora.pdf"}), 0.9)
        ]
        mock_vectorstore.get.return_value = {"ids": ["1"]}

        with patch('app.rag.load_vector_db', return_value=mock_vectorstore):
            with patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'}):
                with patch('app.rag.requests.post', return_value=mock_response):
                    result = rag.ask_rag("Aurora-X7 支持哪些 Wi-Fi 频段？", model_provider="siliconflow")

                    self.assertEqual(result["model_provider"], "siliconflow")
                    self.assertTrue(result["has_answer"])
                    self.assertIn("Wi-Fi", result["answer"])


if __name__ == "__main__":
    unittest.main()
