from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app.agent import RagAgent, build_rag_tool
from app.config import AppConfig
from app.prompts import PromptRenderer
from app.rag import RagIndex


def _configure_logging() -> None:
    log_path = Path("rag_debug.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stdout_handler, file_handler],
        force=True,
    )

    logging.getLogger("app.agent").setLevel(logging.INFO)
    logging.getLogger("app.rag").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger(__name__).info(
        "Logging configured for Streamlit. Terminal and file output enabled at %s",
        log_path.resolve(),
    )


def _get_agent(config_path: Path) -> RagAgent:
    load_dotenv()
    _configure_logging()
    config = AppConfig.from_json(config_path)
    client = OpenAI()
    prompts = PromptRenderer(config.prompts)
    rag = RagIndex(client=client, config=config.rag)
    rag.ensure_index(force=False)
    rag_tool = build_rag_tool(config=config, rag=rag)
    return RagAgent(
        client=client, config=config, prompts=prompts, rag=rag, tools=[rag_tool]
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        .stMarkdown, .stText, .stChatMessage, .stTextInput, .stTextArea, .stSelectbox {
            color: #ffffff;
        }
        .stChatInput textarea {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
        }
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ffffff !important;
        }
        .stButton > button:hover {
            background-color: #e6e6e6 !important;
            color: #000000 !important;
            border: 1px solid #e6e6e6 !important;
        }
        .stSidebar {
            background-color: #0b0b0b;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant_message(answer_text: str, links: list[dict]) -> None:
    st.markdown(answer_text)
    if not links:
        return

    title = (
        "**\u041a\u043e\u0440\u0438\u0441\u043d\u0435 \u043f\u043e\u0441\u0438\u043b\u0430\u043d\u043d\u044f**"
        if len(links) == 1
        else "**\u041a\u043e\u0440\u0438\u0441\u043d\u0456 \u043f\u043e\u0441\u0438\u043b\u0430\u043d\u043d\u044f**"
    )
    st.markdown(title)
    for link in links:
        label = link.get("title") or "\u0414\u0435\u0442\u0430\u043b\u044c\u043d\u0456\u0448\u0435"
        st.markdown(f"- [{label}]({link['url']})")


def main() -> None:
    st.set_page_config(page_title="PDF RAG Assistant", page_icon="DOC", layout="centered")
    _inject_styles()
    st.title("PDF RAG Assistant")
    st.caption("Chat with your company PDFs.")

    config_path = st.sidebar.text_input("Config path", value="config.json")
    if st.sidebar.button("Reload config"):
        st.session_state.pop("agent", None)

    if "agent" not in st.session_state:
        st.session_state["agent"] = _get_agent(Path(config_path))

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(
                    message["content"],
                    message.get("links", []),
                )
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        agent: RagAgent = st.session_state["agent"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent.answer(question=prompt)
                _render_assistant_message(result.text, result.links)
        st.session_state["messages"].append(
            {"role": "assistant", "content": result.text, "links": result.links}
        )


if __name__ == "__main__":
    main()
