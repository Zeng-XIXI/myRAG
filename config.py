"""集中读取配置：改模型时编辑 .env，不需要改业务代码。"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@dataclass
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    docs_dir: Path = ROOT / "docs"
    db_dir: Path = ROOT / "storage"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("需要满足：0 <= CHUNK_OVERLAP < CHUNK_SIZE。")
        if self.top_k < 1:
            raise ValueError("TOP_K 必须大于 0。")


def load_settings() -> Settings:
    # .env 固定从项目目录读取；终端已设置的环境变量优先。
    load_dotenv(ROOT / ".env")
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip().rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
        docs_dir=ROOT / os.getenv("DOCS_DIR", "docs"),
        db_dir=ROOT / os.getenv("DB_DIR", "storage"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "3")),
    )
