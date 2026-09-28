"""无需 API 密钥的行为测试：模型使用替身，向量库使用真实本地 Chroma。"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config import Settings
from models import Models
from rag import COLLECTION, RAG, split_text


class RAGTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.settings = Settings(
            llm_base_url="https://example.invalid/v1", llm_api_key="test",
            llm_model="chat-test", embedding_base_url="https://example.invalid/v1",
            embedding_api_key="test", embedding_model="embedding-test",
            docs_dir=root / "docs", db_dir=root / "storage", top_k=1,
        )
        self.settings.docs_dir.mkdir()
        model_patch = patch("rag.Models", autospec=True)
        self.models = model_patch.start().return_value
        self.addCleanup(model_patch.stop)
        # 人工向量只用于验证检索流程：backup 和其他内容位于不同方向。
        self.models.embed.side_effect = lambda texts: [
            [1.0, 0.0] if "backup" in text else [0.0, 1.0] for text in texts
        ]
        self.rag = RAG(self.settings)

    def write_doc(self, name, text):
        path = self.settings.docs_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def collection(self):
        return self.rag.db.get_collection(COLLECTION, embedding_function=None)

    def test_incremental_sync_adds_and_deletes_only_changed_files(self):
        self.write_doc("keep.md", "保留的资料")
        removed = self.write_doc("remove.txt", "即将删除的资料")
        first = self.rag.ingest()
        self.assertEqual((first.added_files, first.embedded_chunks), (2, 2))

        self.models.embed.reset_mock()
        unchanged = self.rag.ingest()
        self.assertEqual((unchanged.unchanged_files, unchanged.embedded_chunks), (2, 0))
        self.models.embed.assert_not_called()
        self.assertEqual(self.collection().count(), 2)

        removed.unlink()
        self.write_doc("new.md", "新资料")
        changed = self.rag.ingest()
        self.assertEqual(
            (changed.added_files, changed.deleted_files, changed.unchanged_files),
            (1, 1, 1),
        )
        self.models.embed.assert_called_once_with(["新资料"])
        # 新建 RAG 实例，证明记录从同一个持久化目录读取。
        reopened = RAG(self.settings)
        records = reopened.db.get_collection(COLLECTION, embedding_function=None).get()
        self.assertEqual(len(records["ids"]), 2)
        self.assertEqual({item["source"] for item in records["metadatas"]},
                         {"keep.md", "new.md"})

    def test_modified_file_replaces_all_its_chunks(self):
        self.settings.chunk_size = 4
        self.settings.chunk_overlap = 1
        path = self.write_doc("notes.md", "abcdefg")
        self.write_doc("keep.md", "保持不变")
        self.rag.ingest()

        self.models.embed.reset_mock()
        path.write_text("XYZ", encoding="utf-8")
        result = self.rag.ingest()

        self.assertEqual((result.modified_files, result.unchanged_files), (1, 1))
        self.models.embed.assert_called_once_with(["XYZ"])
        records = self.collection().get(include=["documents", "metadatas"])
        notes = [text for text, metadata in zip(
            records["documents"], records["metadatas"]
        ) if metadata["source"] == "notes.md"]
        self.assertEqual(notes, ["XYZ"])

    def test_chunks_store_original_file_positions(self):
        self.write_doc("notes.md", "\nalpha\nbeta\n")
        self.rag.ingest()

        records = self.collection().get(include=["documents", "metadatas"])
        self.assertEqual(records["documents"], ["alpha\nbeta"])
        self.assertEqual(
            {
                key: records["metadatas"][0][key]
                for key in ("start_char", "end_char", "start_line", "end_line")
            },
            {"start_char": 1, "end_char": 11, "start_line": 2, "end_line": 3},
        )

    def test_embedding_failure_preserves_previous_index(self):
        self.write_doc("notes.md", "原有资料")
        self.rag.ingest()
        self.write_doc("notes.md", "替换后的资料")
        self.models.embed.side_effect = RuntimeError("模拟 API 故障")

        with self.assertRaisesRegex(RuntimeError, "模拟 API 故障"):
            self.rag.ingest()
        self.assertEqual(self.collection().get()["documents"], ["原有资料"])

    def test_switching_embedding_model_or_endpoint_requires_reindex(self):
        self.write_doc("notes.md", "已入库资料")
        self.rag.ingest()
        # 替身仍然产生二维向量；模型身份变化也必须拒绝混用。
        for change in ({"embedding_model": "another-two-dimensional-model"},
                       {"embedding_base_url": "https://other.invalid/v1"}):
            with self.subTest(change=change):
                self.models.embed.reset_mock()
                changed = RAG(replace(self.settings, **change))
                with self.assertRaisesRegex(ValueError, "重新运行.*ingest"):
                    changed.search("资料")
                self.models.embed.assert_not_called()

    def test_chunk_setting_change_forces_full_rebuild(self):
        self.write_doc("notes.md", "abcdefgh")
        self.rag.ingest()

        changed = RAG(replace(self.settings, chunk_size=4, chunk_overlap=1))
        changed.models = self.models
        self.models.embed.reset_mock()
        result = changed.ingest()

        self.assertTrue(result.rebuilt)
        self.assertEqual(result.total_chunks, 3)
        self.models.embed.assert_called_once_with(["abcd", "defg", "gh"])

    def test_ask_keeps_original_question_and_actual_retrieval_sources(self):
        self.write_doc("backup.md", "backup：在设置里点击导出备份。")
        self.write_doc("sharing.md", "共享：邀请其他用户加入。")
        self.rag.ingest()
        original = "我想保留一份本地副本，但不要开启共享，怎么操作？"
        self.models.chat.side_effect = ["backup", "在设置里点击导出备份。[1]"]
        events = []

        result = self.rag.ask(
            original,
            progress=events.append,
            request_id="test1234",
        )

        self.assertEqual(result["request_id"], "test1234")
        self.assertEqual(result["query"], "backup")
        self.assertEqual(result["answer"], "在设置里点击导出备份。[1]")
        self.assertEqual(result["sources"][0]["source"], "backup.md")
        completed_stages = [
            event["stage"] for event in events
            if event["state"] == "complete"
        ]
        self.assertEqual(completed_stages, [
            "index_check",
            "rewrite",
            "embedding",
            "vector_search",
            "context_build",
            "answer_generation",
        ])
        self.assertTrue(all(
            result["timings"][f"{stage}_ms"] >= 0
            for stage in completed_stages
        ))
        self.assertGreaterEqual(result["timings"]["rag_total_ms"], 0)
        self.assertEqual(self.models.chat.call_args_list[0].args[1], original)
        final_prompt = self.models.chat.call_args_list[1].args[1]
        self.assertIn(original, final_prompt)
        self.assertIn("[1] 文件：backup.md，片段：1", final_prompt)
        self.assertIn(result["sources"][0]["text"], final_prompt)
        self.assertNotIn("sharing.md", final_prompt)

    def test_missing_index_and_empty_questions_fail_before_api_calls(self):
        for operation in (self.rag.search, self.rag.ask):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ValueError, "先运行.*ingest"):
                    operation("正常问题")
                with self.assertRaisesRegex(ValueError, "不能为空"):
                    operation(" \n ")
        self.models.chat.assert_not_called()
        self.models.embed.assert_not_called()

    def test_failed_stage_emits_error_event(self):
        events = []

        with self.assertRaisesRegex(ValueError, "先运行.*ingest"):
            self.rag.ask(
                "正常问题",
                progress=events.append,
                request_id="failed01",
            )

        self.assertEqual(
            [(event["stage"], event["state"]) for event in events],
            [("index_check", "start"), ("index_check", "error")],
        )
        self.models.chat.assert_not_called()
        self.models.embed.assert_not_called()

    def test_empty_documents_preserve_previous_index(self):
        path = self.write_doc("notes.md", "原有资料")
        self.rag.ingest()
        path.write_text(" \n ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "原索引未修改"):
            self.rag.ingest()
        self.assertEqual(self.collection().get()["documents"], ["原有资料"])


class TextAndModelTests(unittest.TestCase):
    def test_chunk_boundaries_do_not_create_redundant_tail(self):
        for text, expected in (
            ("", []), ("abcd", ["abcd"]), ("abcde", ["abcd", "de"]),
            ("abcdefg", ["abcd", "defg"]),
        ):
            with self.subTest(text=text):
                self.assertEqual(split_text(text, size=4, overlap=1), expected)
        with self.assertRaises(ValueError):
            split_text("abc", size=4, overlap=4)

    def test_embedding_response_is_reordered_by_index(self):
        # 模拟服务商乱序返回；不创建网络连接。
        settings = Settings("", "", "", "https://example.invalid/v1", "test", "embed")
        with patch("models.OpenAI") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.embeddings.create.return_value = SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ])
            self.assertEqual(Models(settings).embed(["第一段", "第二段"]),
                             [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
