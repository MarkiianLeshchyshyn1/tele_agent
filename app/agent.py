from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAI

from app.config import AppConfig
from app.prompts import PromptRenderer
from app.rag import RagIndex
from app.tools import Tool

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
STOP_WORDS = {
    "\u044f\u043a",
    "\u0449\u043e",
    "\u0446\u0435",
    "\u0434\u043b\u044f",
    "\u043f\u0440\u043e",
    "\u0434\u043e",
    "\u0442\u0430",
    "\u0456",
    "\u0439",
    "\u0430\u0431\u043e",
    "\u0447\u0438",
    "\u0432",
    "\u0443",
    "\u043d\u0430",
    "\u0437",
    "\u043f\u043e",
    "\u043d\u0435",
    "\u0430",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
}
INTENT_KEYWORDS = {
    "connect": (
        "підключ",
        "подключ",
        "заявк",
        "під'єд",
        "пiдключ",
        "пiд'єд",
    ),
    "disconnect": (
        "відключ",
        "отключ",
        "розір",
        "скасув",
        "припинен",
        "вiдключ",
    ),
    "tariff": ("тариф", "пакет", "варт", "цін", "цiн", "абонплат"),
    "repair": ("відновлен", "авар", "ремонт", "полом", "не працю", "нема інтернет"),
    "business": ("бізнес", "бiзнес", "business", "корпорат", "юридич"),
    "account": ("кабінет", "кабiнет", "логін", "логiн", "парол", "авторизац"),
}
CONFLICTING_INTENTS = {
    "connect": {"disconnect", "repair"},
    "disconnect": {"connect"},
    "repair": {"connect", "tariff"},
    "tariff": {"repair"},
}


@dataclass(frozen=True)
class AnswerResult:
    text: str
    links: list[dict]
    retrieved_chunks: list[dict]


@dataclass
class RagAgent:
    client: OpenAI
    config: AppConfig
    prompts: PromptRenderer
    rag: RagIndex
    tools: list[Tool]

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        system_prompt = self.prompts.render_system() + _strict_answering_suffix()
        user_prompt = self.prompts.render_user(question)
        retrieval_payload: dict = {"snippets": [], "links": []}
        original_query = question.strip()
        rewritten_queries: list[str] = []
        final_queries: list[str] = []
        retrieval_log_count = 0

        logger.info("Answering user query: original_query=%r", original_query)

        response = self.client.responses.create(
            model=self.config.model,
            input=user_prompt,
            tools=self._tool_schemas(),
            instructions=system_prompt,
            temperature=0,
        )

        input_items = list(response.output)

        while True:
            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            if not tool_calls:
                clean_text = _remove_urls_from_text(
                    response.output_text.strip(),
                    retrieval_payload.get("links", []),
                )
                logger.info(
                    "Final answer for original_query=%r: %r",
                    original_query,
                    clean_text,
                )
                logger.debug(
                    "Final answer debug for original_query=%r: final_queries=%s retrieved_chunks=%s final_answer=%r",
                    original_query,
                    final_queries,
                    retrieval_payload.get("snippets", []),
                    clean_text,
                )
                return AnswerResult(
                    text=clean_text,
                    links=retrieval_payload.get("links", []),
                    retrieved_chunks=retrieval_payload.get("snippets", []),
                )

            for tool_call in tool_calls:
                tool = self._find_tool(tool_call.name)
                if tool is None:
                    continue
                args_obj = json.loads(tool_call.arguments or "{}")
                if tool_call.name == self.config.tool.name:
                    raw_query = args_obj.get("query", "")
                    rewritten_queries.append(raw_query)
                    final_query = _intent_preserving_query(
                        original_query=original_query,
                        requested_query=raw_query,
                    )
                    final_queries.append(final_query)
                    args_obj["query"] = final_query
                    logger.debug(
                        "RAG query debug: original_query=%r raw_tool_query=%r final_query=%r",
                        original_query,
                        raw_query,
                        final_query,
                    )
                if top_k is not None and "top_k" not in args_obj:
                    args_obj["top_k"] = top_k
                result = tool.handler(args_obj)
                if tool_call.name == self.config.tool.name:
                    retrieval_payload = result
                    retrieval_log_count += 1
                    logger.info(
                        "Retrieval summary #%s for original_query=%r final_query=%r: %s",
                        retrieval_log_count,
                        original_query,
                        final_queries[-1] if final_queries else "",
                        _summarize_snippets(retrieval_payload.get("snippets", [])),
                    )
                    logger.debug(
                        "Chunks passed to final LLM context for original_query=%r final_query=%r: %s",
                        original_query,
                        final_queries[-1] if final_queries else "",
                        retrieval_payload.get("snippets", []),
                    )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(result, ensure_ascii=True),
                    }
                )

            response = self.client.responses.create(
                model=self.config.model,
                input=input_items,
                tools=self._tool_schemas(),
                instructions=system_prompt,
                temperature=0,
            )
            input_items.extend(response.output)
            logger.info(
                "RAG query summary: original_query=%r rewritten_queries=%s",
                original_query,
                rewritten_queries,
            )
            logger.debug(
                "RAG query summary debug: original_query=%r rewritten_queries=%s final_queries=%s",
                original_query,
                rewritten_queries,
                final_queries,
            )

    def _tool_schemas(self) -> list[dict]:
        return [tool.schema for tool in self.tools]

    def _find_tool(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


def build_rag_tool(config: AppConfig, rag: RagIndex) -> Tool:
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {
                "type": "integer",
                "description": "Number of snippets to return.",
            },
        },
        "required": ["query"],
    }

    def handler(args: dict) -> dict:
        query = args.get("query", "")
        top_k = args.get("top_k", config.rag.top_k)
        chunks = rag.retrieve(query=query, top_k=top_k)

        snippets = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "url": chunk.url,
                "link": chunk.url,
                "title": chunk.title,
            }
            for chunk in chunks
        ]

        selected_links = _select_relevant_links(query=query, chunks=chunks)

        logger.info(
            "Selected %s relevant links for query %r: %s",
            len(selected_links),
            query,
            selected_links,
        )
        logger.debug(
            "Retrieved chunks full payload for query %r: %s",
            query,
            snippets,
        )

        return {
            "snippets": snippets,
            "links": selected_links,
        }

    return Tool(
        name=config.tool.name,
        description=config.tool.description,
        parameters=parameters,
        handler=handler,
    )


def _select_relevant_links(query: str, chunks: list) -> list[dict]:
    ranked_candidates: list[dict] = []
    seen_urls: set[str] = set()
    query_tokens = _tokenize(query)
    query_intents = _detect_intents(query)

    for rank, chunk in enumerate(chunks):
        if not chunk.url or chunk.url in seen_urls:
            continue
        seen_urls.add(chunk.url)

        title = chunk.title or ""
        title_tokens = _tokenize(chunk.title or "")
        url_tokens = _tokenize(chunk.url)
        text_tokens = _tokenize(chunk.text)
        candidate_intents = _detect_intents(" ".join([title, chunk.url, chunk.text]))

        exact_title_overlap = len(query_tokens & title_tokens)
        exact_url_overlap = len(query_tokens & url_tokens)
        exact_text_overlap = len(query_tokens & text_tokens)
        intent_penalty = _intent_penalty(query_intents, candidate_intents)
        title_penalty = _title_penalty(query_intents, title)

        score = 0.0
        score += exact_title_overlap * 4.0
        score += exact_url_overlap * 3.0
        score += min(exact_text_overlap, 3) * 1.5
        score += max(0.0, 1.0 - rank * 0.15)
        score -= intent_penalty * 3.5
        score -= title_penalty

        candidate = {
            "url": chunk.url,
            "title": chunk.title,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "score": round(score, 3),
            "rank": rank,
            "title_overlap": exact_title_overlap,
            "url_overlap": exact_url_overlap,
            "text_overlap": exact_text_overlap,
            "intent_penalty": round(intent_penalty, 3),
            "title_penalty": round(title_penalty, 3),
        }

        ranked_candidates.append(candidate)

    ranked_candidates.sort(
        key=lambda item: (item["score"], -item["rank"], item["title_overlap"]),
        reverse=True,
    )

    logger.info("Ranked link candidates for query %r: %s", query, ranked_candidates)

    if not ranked_candidates:
        return []

    best = ranked_candidates[0]
    if not _is_link_relevant(best):
        return []

    selected = [_public_link(best)]
    if len(ranked_candidates) < 2:
        return selected

    second = ranked_candidates[1]
    if _should_include_second_link(best=best, second=second):
        selected.append(_public_link(second))
    return selected


def _is_link_relevant(candidate: dict) -> bool:
    if candidate.get("intent_penalty", 0.0) > 0.0 or candidate.get("title_penalty", 0.0) > 0.0:
        return False
    if candidate["title_overlap"] >= 1 or candidate["url_overlap"] >= 1:
        return True
    return candidate["score"] >= 4.0 and candidate["text_overlap"] >= 2


def _should_include_second_link(best: dict, second: dict) -> bool:
    if not _is_link_relevant(second):
        return False
    if second["score"] < best["score"] * 0.8:
        return False
    if second["title"] and best["title"] and second["title"] == best["title"]:
        return False
    return True


def _public_link(candidate: dict) -> dict:
    return {
        "url": candidate["url"],
        "title": candidate["title"],
        "source": candidate["source"],
        "chunk_id": candidate["chunk_id"],
        "page": candidate["page"],
    }


def _tokenize(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token.lower() not in STOP_WORDS
    }
    return tokens


def _intent_preserving_query(original_query: str, requested_query: str) -> str:
    if not original_query:
        return requested_query
    return original_query


def _detect_intents(text: str) -> set[str]:
    normalized = text.lower()
    intents: set[str] = set()
    for intent, markers in INTENT_KEYWORDS.items():
        if any(marker in normalized for marker in markers):
            intents.add(intent)
    return intents


def _intent_penalty(query_intents: set[str], candidate_intents: set[str]) -> float:
    penalty = 0.0
    for intent in query_intents:
        conflicts = CONFLICTING_INTENTS.get(intent, set())
        if candidate_intents & conflicts:
            penalty += 1.0
    if "business" in candidate_intents and "business" not in query_intents:
        penalty += 1.0
    return penalty


def _title_penalty(query_intents: set[str], title: str) -> float:
    normalized_title = (title or "").lower()
    penalty = 0.0
    if "business" not in query_intents and ("для бізнесу" in normalized_title or "business" in normalized_title):
        penalty += 3.0
    if "repair" not in query_intents and "відновлення інтернету" in normalized_title:
        penalty += 3.0
    if "disconnect" not in query_intents and "відключ" in normalized_title:
        penalty += 2.5
    return penalty


def _remove_urls_from_text(text: str, links: list[dict]) -> str:
    if not text:
        return text

    cleaned = text
    for link in links:
        url = link.get("url")
        if url:
            cleaned = cleaned.replace(url, "")

    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip(" -:\t")
        if not line:
            continue
        if line.lower() in {"useful links", "useful link", "корисні посилання", "корисне посилання"}:
            continue
        lines.append(line)

    return "\n\n".join(lines).strip()


def _strict_answering_suffix() -> str:
    return """

Strict answer rules:
- Answer only from retrieved snippets returned by the tool.
- Do not add assumptions, typical steps, guessed prices, guessed conditions, or guessed availability.
- If the retrieved snippets are insufficient, explicitly say that the documents do not contain enough information.
- Prefer quoting or closely paraphrasing the retrieved snippets instead of generalizing.
- If a detail is not present in retrieved snippets, do not fill it in from prior knowledge.
- Keep the answer concise, factual, and grounded in the retrieved context only.
""".strip()


def _summarize_snippets(snippets: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": snippet.get("chunk_id"),
            "page": snippet.get("page"),
            "title": snippet.get("title"),
            "url": snippet.get("url"),
            "preview": _preview_text(snippet.get("text", "")),
        }
        for snippet in snippets
    ]


def _preview_text(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
