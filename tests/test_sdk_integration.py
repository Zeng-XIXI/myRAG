"""真实 SDK + Chroma 的完整流程；MockTransport 在内存中接收所有 HTTP 请求。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from config import Settings
from rag import RAG


class SDKIntegrationTests(unittest.TestCase):
    def test_ingest_and_ask_with_separate_model_services(self):
        requests = []
        replies = iter(["  backup  ", "  在设置中导出备份。[1]  "])

        def handle(request):
            self.assertEqual(request.method, "POST")
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/v1/embeddings":
                self.assertEqual(request.url.host, "vectors.example.invalid")
                self.assertEqual(request.headers["authorization"], "Bearer vector-key")
                self.assertEqual(body["model"], "vector-model")
                self.assertEqual(body["encoding_format"], "float")
                self.assertIsInstance(body["input"], list)
                # 故意乱序，检索正确来源也依赖 SDK 解析后的 index 重排。
                data = [
                    {"object": "embedding", "index": index,
                     "embedding": [1.0, 0.0] if "backup" in text else [0.0, 1.0]}
                    for index, text in enumerate(body["input"])
                ]
                response = {"object": "list", "model": "vector-model",
                            "data": list(reversed(data)),
                            "usage": {"prompt_tokens": 10, "total_tokens": 10}}
            else:
                self.assertEqual(request.url.path, "/v1/chat/completions")
                self.assertEqual(request.url.host, "chat.example.invalid")
                self.assertEqual(request.headers["authorization"], "Bearer chat-key")
                self.assertEqual(body["model"], "chat-model")
                self.assertEqual([item["role"] for item in body["messages"]],
                                 ["system", "user"])
                response = {
                    "id": "chat-test", "object": "chat.completion", "created": 1,
                    "model": "chat-model", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": next(replies)},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15},
                }
            return httpx.Response(200, json=response)

        def client_with_mock_transport(**kwargs):
            # 只注入 HTTP transport；SDK 的请求序列化和响应解析均保留。
            return OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(handle)),
                          **kwargs)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "backup.md").write_text("backup：在设置中导出备份。", encoding="utf-8")
            (docs / "sharing.md").write_text("共享：邀请成员。", encoding="utf-8")
            settings = Settings(
                llm_base_url="https://chat.example.invalid/v1", llm_api_key="chat-key",
                llm_model="chat-model", embedding_base_url="https://vectors.example.invalid/v1",
                embedding_api_key="vector-key", embedding_model="vector-model",
                docs_dir=docs, db_dir=root / "storage", top_k=1,
            )
            with patch("models.OpenAI", side_effect=client_with_mock_transport):
                rag = RAG(settings)
                ingest_result = rag.ingest()
                self.assertEqual(
                    (ingest_result.added_files, ingest_result.total_chunks), (2, 2)
                )
                result = rag.ask("如何备份？")

        self.assertEqual([path for path, _ in requests], [
            "/v1/embeddings", "/v1/chat/completions",
            "/v1/embeddings", "/v1/chat/completions",
        ])
        self.assertEqual(requests[1][1]["messages"][1]["content"], "如何备份？")
        self.assertEqual(requests[2][1]["input"], ["backup"])
        self.assertIn("backup.md", requests[3][1]["messages"][1]["content"])
        self.assertEqual(result["query"], "backup")
        self.assertEqual(result["answer"], "在设置中导出备份。[1]")
        self.assertEqual(result["sources"][0]["source"], "backup.md")


if __name__ == "__main__":
    unittest.main()
