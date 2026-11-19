"""LLM-шаги вокруг детерминированного ядра.

write-path:  фрагмент диалога → классификатор разделов → extractor по каждому разделу → MemoryWriter
read-path:   сообщение → классификатор запроса (fallback: правила) → ContextBuilder
ответ:       контекст + история → основная модель

Граница ответственности: всё, что вернула модель, проходит через код.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from .compliance import ComplianceChecker, PostCheck
from .followup import ClarifyCheck, clarify_check
from .lexicon import fold
from .llm import LLM, Messages, extract_json_object
from .memory import MemoryWriter, WriteReport
from .normalize import Normalizer
from .prompts import (
    answer_messages,
    compact_dialog,
    extractor_messages,
    query_classifier_messages,
    section_classifier_messages,
)
from .query_rules import classify_query_rules
from .retrieval import ContextBuilder, ContextBundle, DocumentRecord
from .schemas import QueryClassification, QueryType, parse_extraction
from .sections import Section
from .store import MemoryStore

Turns = list[dict[str, str]]


@dataclass
class LLMCall:
    task: str
    section: str | None
    raw: str
    ok: bool
    error: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    parsed: Any = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


REPAIR_PROMPT = ("Ответ не удалось разобрать как JSON ({error}). Верни только JSON-объект в требуемом формате, "
                 "без пояснений и без markdown.")


def json_call(
    llm: LLM,
    messages: Messages,
    task: str,
    section: str | None = None,
    max_new_tokens: int = 512,
    retries: int = 1,
) -> tuple[dict | None, LLMCall]:
    msgs = list(messages)
    raw, error, latency = "", None, 0.0
    for attempt in range(1, retries + 2):
        tokens = max_new_tokens if attempt == 1 else int(max_new_tokens * 1.5)
        t0 = time.perf_counter()
        raw = llm.chat(msgs, max_new_tokens=tokens, json_mode=True,
                       meta={"task": task, "section": section, "attempt": attempt})
        latency += time.perf_counter() - t0
        try:
            obj = extract_json_object(raw)
            return obj, LLMCall(task, section, raw, ok=True, attempts=attempt, latency_s=latency)
        except ValueError as exc:
            error = str(exc)
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": REPAIR_PROMPT.format(error=error)}]
    return None, LLMCall(task, section, raw, ok=False, error=error, attempts=retries + 1, latency_s=latency)


# --------------------------------------------------------------------------- write-path


class SectionClassifier:
    def __init__(self, llm: LLM, max_new_tokens: int = 64):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def __call__(self, turns: Turns) -> tuple[list[Section], LLMCall]:
        obj, call = json_call(self.llm, section_classifier_messages(turns), "classify_sections",
                              max_new_tokens=self.max_new_tokens)
        if obj is None:
            return [], call
        raw = obj.get("sections")
        if not isinstance(raw, list):
            call.ok, call.error = False, "нет списка sections"
            return [], call
        valid: list[Section] = []
        unknown: list[Any] = []
        for item in raw:
            try:
                sec = Section(item)
            except ValueError:
                unknown.append(item)
                continue
            if sec not in valid:
                valid.append(sec)
        if unknown:
            call.error = f"неизвестные разделы отброшены: {unknown}"
        call.parsed = [s.value for s in valid]
        return valid, call


@dataclass
class ExtractionStats:
    off_section: int = 0
    failed_sections: int = 0
    repairs: int = 0
    repaired: int = 0


VALIDATION_REPAIR_PROMPT = """Часть элементов не прошла проверку:
{errors}
Исправь эти элементы по правилам раздела или удали их, если это не сведения раздела. Верни весь JSON заново, без пояснений."""


def _validation_errors(facts: list, statuses: list) -> tuple[int, list[str]]:
    valid_f, valid_s, rejected = parse_extraction({"facts": facts, "section_statuses": statuses})
    lines = [f"- {json.dumps(r.raw, ensure_ascii=False)[:200]} → {'; '.join(r.errors)}" for r in rejected]
    return len(valid_f) + len(valid_s), lines


class Extractor:
    def __init__(self, llm: LLM, max_new_tokens: int = 512, repair: bool = True):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.repair = repair

    @staticmethod
    def _collect(obj: dict, section: Section, call: LLMCall) -> tuple[list, list, int]:
        out: dict[str, list] = {"facts": [], "section_statuses": []}
        off = 0
        for key in ("facts", "section_statuses"):
            items = obj.get(key) or []
            if not isinstance(items, list):
                call.error = f"{key}: ожидается список"
                continue
            for item in items:
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("section", section.value)
                    if item["section"] != section.value:
                        off += 1
                        continue
                out[key].append(item)
        return out["facts"], out["section_statuses"], off

    def __call__(self, turns: Turns, sections: Iterable[Section]) -> tuple[dict, list[LLMCall], ExtractionStats]:
        merged: dict[str, list] = {"facts": [], "section_statuses": []}
        calls: list[LLMCall] = []
        stats = ExtractionStats()
        for section in sections:
            messages = extractor_messages(section, turns)
            obj, call = json_call(self.llm, messages, "extract", section.value, max_new_tokens=self.max_new_tokens)
            calls.append(call)
            if obj is None:
                stats.failed_sections += 1
                continue
            facts, statuses, off = self._collect(obj, section, call)
            stats.off_section += off
            n_valid, errors = _validation_errors(facts, statuses)
            if self.repair and errors:
                stats.repairs += 1
                retry_msgs = messages + [
                    {"role": "assistant", "content": call.raw},
                    {"role": "user", "content": VALIDATION_REPAIR_PROMPT.format(errors="\n".join(errors))},
                ]
                obj2, call2 = json_call(self.llm, retry_msgs, "extract_repair", section.value,
                                        max_new_tokens=self.max_new_tokens)
                calls.append(call2)
                if obj2 is not None:
                    f2, s2, _ = self._collect(obj2, section, call2)
                    n_valid2, errors2 = _validation_errors(f2, s2)
                    if n_valid2 >= n_valid and len(errors2) < len(errors):
                        facts, statuses = f2, s2
                        stats.repaired += 1
                    call2.parsed = {"valid": n_valid2, "errors": len(errors2), "accepted": facts is f2}
            call.parsed = len(facts) + len(statuses)
            merged["facts"] += facts
            merged["section_statuses"] += statuses
        return merged, calls, stats


@dataclass
class WriteResult:
    dialog_id: str | None
    sections: list[Section]
    extraction: dict
    report: WriteReport
    calls: list[LLMCall]
    stats: ExtractionStats
    classifier_failed: bool = False
    turns: Turns = field(default_factory=list)
    classifier_sections: list[Section] = field(default_factory=list)
    safety_net: list[Section] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "dialog_id": self.dialog_id,
            "sections": [s.value for s in self.sections],
            "safety_net": [s.value for s in self.safety_net],
            "changes": self.report.counts(),
            "rejected": len(self.report.rejected),
            "off_section": self.stats.off_section,
            "failed_sections": self.stats.failed_sections,
            "repairs": self.stats.repairs,
            "repaired": self.stats.repaired,
            "classifier_failed": self.classifier_failed,
            "llm_calls": len(self.calls),
            "retries": sum(c.attempts - 1 for c in self.calls),
            "latency_s": round(sum(c.latency_s for c in self.calls), 2),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "turns": self.turns,
            "extraction": self.extraction,
            "report": self.report.to_rows(),
            "calls": [c.as_dict() for c in self.calls],
        }


class WritePath:
    def __init__(self, llm: LLM, writer: MemoryWriter,
                 classifier: SectionClassifier | None = None, extractor: Extractor | None = None,
                 safety_net: bool = True):
        self.writer = writer
        self.classifier = classifier or SectionClassifier(llm)
        self.extractor = extractor or Extractor(llm)
        self.use_safety_net = safety_net

    CP_ABSENT_RE = re.compile(
        r"(контрагент|поставщик|покупател)\w*\s+(\w+\s+){0,2}нет"
        r"|нет\s+(\w+\s+){0,2}(контрагент|поставщик|покупател)"
        r"|ни\s+с\s+кем\s+не\s+работа"
    )

    def safety_net(self, turns: Turns, sections: list[Section]) -> list[Section]:
        """Разделы, которые классификатор пропустил, а код видит по справочнику или шаблону."""
        n = self.writer.normalizer
        text = "\n".join(t["content"] for t in turns)
        added: list[Section] = []
        if Section.COUNTERPARTIES not in sections and (
                n.counterparty_mentions(text) or self.CP_ABSENT_RE.search(fold(text))):
            added.append(Section.COUNTERPARTIES)
        if Section.OBLIGATIONS not in sections and n.obligations.find_mentions(text):
            added.append(Section.OBLIGATIONS)
        return added

    def process(self, user_id: str, turns: Turns, dialog_id: str | None = None) -> WriteResult:
        turns = compact_dialog(turns)
        sections, ccall = self.classifier(turns)
        classifier_sections = list(sections)
        classifier_failed = not ccall.ok
        added: list[Section] = []
        if classifier_failed:
            sections = list(Section)
        elif self.use_safety_net:
            added = self.safety_net(turns, sections)
            sections = sections + added
        extraction, ecalls, stats = self.extractor(turns, sections) if sections else (
            {"facts": [], "section_statuses": []}, [], ExtractionStats())
        report = self.writer.write(user_id, extraction, dialog_id,
                                   dialog_text="\n".join(t["content"] for t in turns))
        failed = [c for c in [ccall, *ecalls] if not c.ok]
        if failed:
            now = self.writer.clock()
            with self.writer.store.transaction():
                for c in failed:
                    self.writer.store.add_rejection(
                        user_id, dialog_id, now, "output",
                        {"task": c.task, "section": c.section, "raw": c.raw[:2000]}, [c.error or "ошибка"],
                    )
        return WriteResult(dialog_id, sections, extraction, report, [ccall, *ecalls], stats, classifier_failed,
                           turns, classifier_sections, added)


# --------------------------------------------------------------------------- read-path


class QueryClassifier:
    def __init__(self, llm: LLM | None, fallback: Callable[[str], QueryClassification] = classify_query_rules,
                 max_new_tokens: int = 64):
        self.llm = llm
        self.fallback = fallback
        self.max_new_tokens = max_new_tokens

    def __call__(self, message: str, history: Turns | None = None) -> tuple[QueryClassification, LLMCall | None, bool]:
        if self.llm is None:
            return self.fallback(message), None, True
        obj, call = json_call(self.llm, query_classifier_messages(message, history), "classify_query",
                              max_new_tokens=self.max_new_tokens)
        if obj is not None:
            raw_types = obj.get("query_types")
            if isinstance(raw_types, str):
                raw_types = [raw_types]
            known = {q.value for q in QueryType}
            types = [t for t in (raw_types or []) if t in known]
            dropped = [t for t in (raw_types or []) if t not in known]
            if dropped:
                call.error = f"неизвестные типы отброшены: {dropped}"
            try:
                qc = QueryClassification(query_types=types, subject=obj.get("subject") or "self")
                call.parsed = qc.model_dump(mode="json")
                return qc, call, False
            except ValidationError as exc:
                call.ok = False
                call.error = f"валидация: {exc.errors()[0]['msg']}"
        return self.fallback(message), call, True


@dataclass
class ReadResult:
    classification: QueryClassification
    fallback_used: bool
    bundle: ContextBundle
    call: LLMCall | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_types": [q.value for q in self.classification.query_types],
            "subject": self.classification.subject,
            "fallback_used": self.fallback_used,
            "sections": [s.value for s in self.bundle.sections],
            "clarify": [c.render() for c in self.bundle.clarify],
            "context": self.bundle.render(),
            "compliance": self.bundle.compliance.as_dict() if self.bundle.compliance else None,
            "call": self.call.as_dict() if self.call else None,
        }


class ReadPath:
    def __init__(self, builder: ContextBuilder, query_classifier: QueryClassifier):
        self.builder = builder
        self.classifier = query_classifier

    def __call__(self, user_id: str, message: str, history: Turns | None = None,
                 documents: Iterable[DocumentRecord] | None = None) -> ReadResult:
        qc, call, fallback = self.classifier(message, history)
        said = "\n".join([t["content"] for t in (history or []) if t["role"] == "user"] + [message])
        bundle = self.builder.build(user_id, qc.query_types, qc.subject, documents=documents, reference_text=said)
        return ReadResult(qc, fallback, bundle, call)


# --------------------------------------------------------------------------- ассистент и сессия


@dataclass
class AssistantReply:
    text: str
    read: ReadResult
    latency_s: float
    raw_text: str = ""
    compliance: PostCheck | None = None
    clarify: ClarifyCheck | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "raw_text": self.raw_text, "latency_s": round(self.latency_s, 2),
                "compliance": self.compliance.as_dict() if self.compliance else None,
                "clarify": self.clarify.as_dict() if self.clarify else None, "read": self.read.as_dict()}


class AccountantAssistant:
    """Собирает всё вместе. Память — одна на ассистента, компании разделены user_id.

    compliance=True (по умолчанию) — проверка по справочнику до и после ответа модели.
    """

    def __init__(
        self,
        llm: LLM,
        store: MemoryStore | None = None,
        normalizer: Normalizer | None = None,
        clock: Callable[[], datetime] | None = None,
        use_llm_query_classifier: bool = True,
        answer_max_new_tokens: int = 512,
        compliance: ComplianceChecker | bool = True,
    ):
        self.llm = llm
        self.store = store or MemoryStore()
        self.normalizer = normalizer or Normalizer()
        self.compliance: ComplianceChecker | None = (
            ComplianceChecker(self.normalizer.catalog) if compliance is True else compliance or None)
        self.writer = MemoryWriter(self.store, self.normalizer, clock)
        self.builder = ContextBuilder(self.store, self.normalizer, self.writer.clock, compliance=self.compliance)
        self.write_path = WritePath(llm, self.writer)
        self.read_path = ReadPath(self.builder, QueryClassifier(llm if use_llm_query_classifier else None))
        self.answer_max_new_tokens = answer_max_new_tokens

    def reply(self, user_id: str, message: str, history: Turns | None = None,
              documents: Iterable[DocumentRecord] | None = None) -> AssistantReply:
        history = history or []
        read = self.read_path(user_id, message, history, documents)
        t0 = time.perf_counter()
        raw = self.llm.chat(answer_messages(read.bundle.render(), history, message),
                            max_new_tokens=self.answer_max_new_tokens, meta={"task": "answer"}).strip()
        post = clar = None
        text = raw
        if self.compliance is not None and read.bundle.compliance is not None:
            post = self.compliance.post_check(raw, read.bundle.compliance.profile,
                                              read.bundle.compliance.question_entities)
            clar = clarify_check(post.text, read.bundle.clarify, read.bundle.subject, bool(post.answer_entities))
            text = clar.text
        return AssistantReply(text, read, time.perf_counter() - t0, raw, post, clar)

    def session(self, user_id: str, dialog_id: str | None = None,
                documents: Iterable[DocumentRecord] | None = None) -> "ChatSession":
        return ChatSession(self, user_id, dialog_id, list(documents or []))


@dataclass
class ChatSession:
    """Read-path — на каждое сообщение; write-path — после фрагмента диалога (`close`)."""

    assistant: AccountantAssistant
    user_id: str
    dialog_id: str | None = None
    documents: list[DocumentRecord] = field(default_factory=list)
    turns: Turns = field(default_factory=list)
    replies: list[AssistantReply] = field(default_factory=list)

    def say(self, message: str) -> str:
        reply = self.assistant.reply(self.user_id, message, self.turns, self.documents)
        self.turns += [{"role": "user", "content": message}, {"role": "assistant", "content": reply.text}]
        self.replies.append(reply)
        return reply.text

    def close(self) -> WriteResult:
        return self.assistant.write_path.process(self.user_id, self.turns, self.dialog_id)
