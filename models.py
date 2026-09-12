"""唯一接触模型 API 的文件；不兼容 OpenAI 的服务也只需改这里。"""

from openai import OpenAI

from config import Settings


class Models:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _client(self, kind: str) -> OpenAI:
        # 按需创建：入库只需要向量服务，不要求聊天服务也配置好。
        base_url = getattr(self.settings, f"{kind}_base_url")
        api_key = getattr(self.settings, f"{kind}_api_key")
        model = getattr(self.settings, f"{kind}_model")
        if not all((base_url, api_key, model)):
            raise ValueError(f"请在 .env 中填写完整的 {kind.upper()} 配置。")
        return OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=1)

    def chat(self, system: str, user: str) -> str:
        """输入指令和问题，返回模型生成的文字。"""
        with self._client("llm") as client:
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        if not response.choices:
            raise RuntimeError("聊天模型没有返回回答，请检查模型配置。")
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("模型输出被截断，请检查服务的输出长度限制。")
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise RuntimeError("聊天模型返回了空内容，请检查模型是否支持文本对话。")
        return text.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """一段文字对应一个向量；入库和提问都使用此函数。"""
        vectors = []
        with self._client("embedding") as client:
            # 小批量请求，避免一次提交整份资料；服务商限额不同时可改这里。
            for start in range(0, len(texts), 10):
                batch = texts[start:start + 10]
                response = client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                    encoding_format="float",
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in ordered] != list(range(len(batch))):
                    raise RuntimeError("向量接口返回的数量或顺序标识有误，已停止入库/检索。")
                vectors.extend(item.embedding for item in ordered)
        return vectors
