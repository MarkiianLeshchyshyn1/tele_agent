from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptConfig:
    prompt_dir: Path
    system_prompt_file: str
    user_prompt_file: str

    @property
    def system_prompt_path(self) -> Path:
        return self.prompt_dir / self.system_prompt_file

    @property
    def user_prompt_path(self) -> Path:
        return self.prompt_dir / self.user_prompt_file


@dataclass(frozen=True)
class RagConfig:
    data_dir: Path
    index_filename: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str
    embedding_batch_size: int

    @property
    def index_path(self) -> Path:
        return self.data_dir / self.index_filename


@dataclass(frozen=True)
class ToolConfig:
    name: str
    description: str


@dataclass(frozen=True)
class AppConfig:
    model: str
    prompts: PromptConfig
    rag: RagConfig
    tool: ToolConfig

    @classmethod
    def from_json(cls, path: Path | str) -> "AppConfig":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        prompts = PromptConfig(
            prompt_dir=Path(payload["prompt_dir"]),
            system_prompt_file=payload["system_prompt_file"],
            user_prompt_file=payload["user_prompt_file"],
        )

        rag = RagConfig(
            data_dir=Path(payload["data_dir"]),
            index_filename=payload["index_filename"],
            chunk_size=int(payload["chunk_size"]),
            chunk_overlap=int(payload["chunk_overlap"]),
            top_k=int(payload["top_k"]),
            embedding_model=payload["embedding_model"],
            embedding_batch_size=int(payload["embedding_batch_size"]),
        )

        tool = ToolConfig(
            name=payload["tool_name"],
            description=payload["tool_description"],
        )

        return cls(
            model=payload["model"],
            prompts=prompts,
            rag=rag,
            tool=tool,
        )
