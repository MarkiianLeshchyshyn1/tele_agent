from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from pypdf import PdfReader

from app.config import RagConfig

logger = logging.getLogger(__name__)
INDEX_VERSION = 3
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
TITLE_PATTERNS = (
    re.compile(
        r"\u041d\u0430\u0437\u0432\u0430 \u0441\u0442\u043e\u0440\u0456\u043d\u043a\u0438:\s*(.+?)\s+URL\s*:",
        re.IGNORECASE,
    ),
    re.compile(r"Title:\s*(.+?)\s+URL\s*:", re.IGNORECASE),
)
STOP_WORDS = {
    "як",
    "що",
    "це",
    "для",
    "про",
    "до",
    "та",
    "і",
    "й",
    "або",
    "чи",
    "в",
    "у",
    "на",
    "з",
    "по",
    "не",
    "а",
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
}
CONFLICTING_INTENTS = {
    "connect": {"disconnect", "repair"},
    "disconnect": {"connect"},
    "repair": {"connect", "tariff"},
    "tariff": {"repair"},
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    url: str | None = None
    title: str | None = None
    score: float | None = None


@dataclass
class RagIndex:
    client: OpenAI
    config: RagConfig
    _index_payload: dict | None = None

    def ensure_index(self, force: bool = False) -> dict:
        if force or self._index_payload is None:
            self._index_payload = self._load_or_build_index(force=force)
        return self._index_payload

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        payload = self.ensure_index()
        top_k = max(1, top_k or self.config.top_k)

        query_embedding = self._embed_texts([query])[0]
        query_norm = self._vector_norm(query_embedding)
        query_tokens = _tokenize(query)
        query_intents = _detect_intents(query)

        scored: list[tuple[float, float, dict]] = []
        for chunk in payload.get("chunks", []):
            semantic_score = self._cosine_similarity(
                query_embedding, query_norm, chunk["embedding"], chunk["norm"]
            )
            title = chunk.get("title") or ""
            url = chunk.get("url") or ""
            text = chunk.get("text") or ""
            title_overlap = len(query_tokens & _tokenize(title))
            url_overlap = len(query_tokens & _tokenize(url))
            text_overlap = len(query_tokens & _tokenize(text))
            candidate_intents = _detect_intents(" ".join([title, url, text]))
            intent_penalty = _intent_penalty(query_intents, candidate_intents)
            hybrid_score = semantic_score
            hybrid_score += title_overlap * 0.08
            hybrid_score += url_overlap * 0.06
            hybrid_score += min(text_overlap, 5) * 0.02
            hybrid_score -= intent_penalty * 0.18
            scored.append(
                (
                    hybrid_score,
                    semantic_score,
                    {
                        **chunk,
                        "_debug": {
                            "hybrid_score": round(hybrid_score, 4),
                            "semantic_score": round(semantic_score, 4),
                            "title_overlap": title_overlap,
                            "url_overlap": url_overlap,
                            "text_overlap": text_overlap,
                            "intent_penalty": round(intent_penalty, 3),
                        },
                    },
                )
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results: list[Chunk] = []
        for hybrid_score, semantic_score, chunk in scored[:top_k]:
            results.append(
                Chunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    source=chunk["source"],
                    page=chunk.get("page"),
                    url=chunk.get("url"),
                    title=chunk.get("title"),
                    score=chunk.get("_debug", {}).get("hybrid_score"),
                )
            )

        logger.info(
            "Top retrieved chunk previews for query %r: %s",
            query,
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk.get("page"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "score": chunk.get("_debug", {}).get("hybrid_score"),
                    "text_preview": _preview_text(chunk.get("text", "")),
                }
                for _, _, chunk in scored[:top_k]
            ],
        )
        logger.debug(
            "Retrieved chunks debug for query %r: %s",
            query,
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "debug": chunk.get("_debug", {}),
                    "text": chunk.get("text"),
                }
                for _, _, chunk in scored[:top_k]
            ],
        )
        return results

    def _load_or_build_index(self, force: bool = False) -> dict:
        if force or not self.config.index_path.exists():
            return self._build_index()
        payload = json.loads(self.config.index_path.read_text(encoding="utf-8"))
        if (
            payload.get("embedding_model") != self.config.embedding_model
            or payload.get("chunk_size") != self.config.chunk_size
            or payload.get("chunk_overlap") != self.config.chunk_overlap
            or self._index_is_stale(payload)
        ):
            return self._build_index()
        if payload.get("version") != INDEX_VERSION:
            return self._build_index()
        return payload

    def _build_index(self) -> dict:
        pdf_paths = sorted(p for p in self.config.data_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {self.config.data_dir}")

        texts: list[str] = []
        meta: list[tuple[str, int, str | None, str | None]] = []
        for pdf_path in pdf_paths:
            for page_num, page_text in self._extract_pdf_pages(pdf_path):
                page_url = self._extract_first_url(page_text)
                page_title = self._extract_title(page_text)
                for chunk in self._chunk_text(
                    page_text, self.config.chunk_size, self.config.chunk_overlap
                ):
                    chunk_url = self._extract_first_url(chunk) or page_url
                    chunk_title = self._extract_title(chunk) or page_title
                    texts.append(chunk)
                    meta.append(
                        (
                            pdf_path.name,
                            page_num,
                            chunk_url,
                            chunk_title,
                        )
                    )

        embeddings = self._embed_texts(texts)
        chunks: list[dict] = []
        for idx, ((source, page, url, title), text, embedding) in enumerate(
            zip(meta, texts, embeddings),
            start=1,
        ):
            norm = self._vector_norm(embedding)
            chunks.append(
                {
                    "chunk_id": f"{Path(source).stem}-chunk-{idx}",
                    "text": text,
                    "source": source,
                    "page": page,
                    "url": url,
                    "title": title,
                    "embedding": embedding,
                    "norm": norm,
                }
            )

        index_payload = {
            "version": INDEX_VERSION,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "files": {p.name: p.stat().st_mtime for p in pdf_paths},
            "chunks": chunks,
        }
        self.config.index_path.write_text(
            json.dumps(index_payload, ensure_ascii=True), encoding="utf-8"
        )
        return index_payload

    def _upgrade_index_payload(self, payload: dict) -> dict:
        upgraded_chunks: list[dict] = []
        for idx, chunk in enumerate(payload.get("chunks", []), start=1):
            text = chunk.get("text", "")
            upgraded_chunk = dict(chunk)
            upgraded_chunk["chunk_id"] = chunk.get(
                "chunk_id",
                f"{Path(chunk.get('source', 'chunk')).stem}-chunk-{idx}",
            )
            upgraded_chunk["page"] = chunk.get("page")
            upgraded_chunk["url"] = chunk.get("url") or self._extract_first_url(text)
            upgraded_chunk["title"] = chunk.get("title") or self._extract_title(text)
            upgraded_chunks.append(upgraded_chunk)

        payload["version"] = INDEX_VERSION
        payload["chunks"] = upgraded_chunks
        self.config.index_path.write_text(
            json.dumps(payload, ensure_ascii=True), encoding="utf-8"
        )
        logger.info("Upgraded existing RAG index metadata to version %s.", INDEX_VERSION)
        return payload

    def _index_is_stale(self, index_payload: dict) -> bool:
        files = index_payload.get("files", {})
        for pdf_path in self.config.data_dir.glob("*.pdf"):
            mtime = pdf_path.stat().st_mtime
            if files.get(pdf_path.name) != mtime:
                return True
        for filename in files:
            if not (self.config.data_dir / filename).exists():
                return True
        return False

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []
        if chunk_size <= 0:
            return [text]
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 4)
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _extract_pdf_pages(path: Path) -> Iterable[tuple[int, str]]:
        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            yield idx, text

    @staticmethod
    def _extract_first_url(text: str) -> str | None:
        match = URL_PATTERN.search(text)
        if not match:
            return None
        return match.group(0).rstrip(".,);]")

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for pattern in TITLE_PATTERNS:
            match = pattern.search(text)
            if match:
                title = " ".join(match.group(1).split()).strip(" \"'()[]")
                if title:
                    return title
        return None

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        batch_size = self.config.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.config.embedding_model, input=batch
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    @staticmethod
    def _vector_norm(vec: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vec)) or 1.0

    @staticmethod
    def _cosine_similarity(
        vec: list[float], norm: float, other: list[float], other_norm: float
    ) -> float:
        dot = 0.0
        for a, b in zip(vec, other):
            dot += a * b
        return dot / (norm * other_norm)


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token.lower() not in STOP_WORDS
    }


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


def _preview_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
