from __future__ import annotations

from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader

from app.config import PromptConfig


@dataclass(frozen=True)
class PromptRenderer:
    config: PromptConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_env",
            Environment(loader=FileSystemLoader(str(self.config.prompt_dir))),
        )

    def render_system(self) -> str:
        return self._env.get_template(self.config.system_prompt_file).render().strip()

    def render_user(self, question: str) -> str:
        return (
            self._env.get_template(self.config.user_prompt_file)
            .render(question=question)
            .strip()
        )
