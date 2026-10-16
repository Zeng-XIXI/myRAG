import unittest

from router import classify_question, normalize_question


class RouterTests(unittest.TestCase):
    def test_normalize_question_removes_spaces_and_punctuation(self):
        self.assertEqual(normalize_question(" 你好！ "), "你好")
        self.assertEqual(normalize_question("HELLO?"), "hello")

    def test_greeting_uses_chat_route(self):
        decision = classify_question("你好")

        self.assertEqual(decision.route, "chat")
        self.assertFalse(decision.rewrite)
        self.assertIsNotNone(decision.answer)
        self.assertIn("你好", decision.answer)

    def test_thanks_uses_chat_route(self):
        decision = classify_question("谢谢！")

        self.assertEqual(decision.route, "chat")
        self.assertFalse(decision.rewrite)
        self.assertIsNotNone(decision.answer)

    def test_empty_question_is_unclear(self):
        decision = classify_question("   ")

        self.assertEqual(decision.route, "unclear")
        self.assertFalse(decision.rewrite)
        self.assertIsNotNone(decision.answer)

    def test_ambiguous_question_is_unclear(self):
        decision = classify_question("这个怎么配置？")

        self.assertEqual(decision.route, "unclear")
        self.assertFalse(decision.rewrite)
        self.assertIn("具体对象", decision.answer)

    def test_short_knowledge_query_uses_rag_without_rewrite(self):
        decision = classify_question("如何导出备份？")

        self.assertEqual(decision.route, "rag")
        self.assertFalse(decision.rewrite)
        self.assertIsNone(decision.answer)

    def test_short_keyword_can_still_use_rag(self):
        decision = classify_question("备份")

        self.assertEqual(decision.route, "rag")
        self.assertFalse(decision.rewrite)

    def test_long_question_uses_rag_with_rewrite(self):
        question = (
            "我想保留一份本地的数据副本，但是不希望开启共享功能，"
            "请问应该进入哪个页面以及需要执行哪些具体操作？"
        )

        decision = classify_question(question)

        self.assertEqual(decision.route, "rag")
        self.assertTrue(decision.rewrite)
        self.assertIsNone(decision.answer)


if __name__ == "__main__":
    unittest.main()