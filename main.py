import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.agent import RagAgent, build_rag_tool
from app.config import AppConfig
from app.prompts import PromptRenderer
from app.rag import RagIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-powered PDF assistant.")
    parser.add_argument("--config", default="config.json", help="Path to config file.")
    parser.add_argument("--question", help="User question to answer.")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing PDFs.")
    return parser.parse_args()


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)


def main() -> None:
    load_dotenv()
    _configure_logging()
    args = parse_args()

    config = AppConfig.from_json(Path(args.config))
    client = OpenAI()
    prompts = PromptRenderer(config.prompts)
    rag = RagIndex(client=client, config=config.rag)
    rag.ensure_index(force=args.reindex)

    question = args.question or input("Ask a question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    rag_tool = build_rag_tool(config=config, rag=rag)
    agent = RagAgent(
        client=client, config=config, prompts=prompts, rag=rag, tools=[rag_tool]
    )
    result = agent.answer(question=question)
    print(result.text)
    if result.links:
        block_title = (
            "\n\u041a\u043e\u0440\u0438\u0441\u043d\u0435 \u043f\u043e\u0441\u0438\u043b\u0430\u043d\u043d\u044f:"
            if len(result.links) == 1
            else "\n\u041a\u043e\u0440\u0438\u0441\u043d\u0456 \u043f\u043e\u0441\u0438\u043b\u0430\u043d\u043d\u044f:"
        )
        print(block_title)
        for link in result.links:
            label = link.get("title") or "\u0414\u0435\u0442\u0430\u043b\u044c\u043d\u0456\u0448\u0435"
            print(f"- {label}: {link['url']}")


if __name__ == "__main__":
    main()
