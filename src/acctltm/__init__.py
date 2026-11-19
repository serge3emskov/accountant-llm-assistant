"""acctltm — долгосрочная память AI-ассистента бухгалтера.

LLM извлекает смысл, состоянием памяти управляет код.
"""
from .catalog import AccountingCatalog, EntityInfo
from .compliance import ComplianceChecker, ComplianceReport, PostCheck, Profile, RuleSet
from .lexicon import Lexicon, NormResult, clean_name, fold, parse_amount, valid_inn
from .llm import CachedLLM, LLM, OpenAICompatLLM, QwenTransformersLLM, ScriptedLLM
from .memory import ChangeType, MemoryWriter, WriteReport
from .normalize import Normalizer
from .pipeline import (
    AccountantAssistant,
    AssistantReply,
    ChatSession,
    Extractor,
    ReadPath,
    ReadResult,
    SectionClassifier,
    WritePath,
    WriteResult,
)
from .query_rules import classify_query_rules
from .retrieval import ContextBuilder, ContextBundle, DocumentRecord
from .schemas import QueryClassification, QueryType, parse_extraction
from .sections import POLICIES, FactKind, Section, SectionPolicy
from .store import EventRecord, Fact, MemoryStore, RejectionRecord, SectionState

__version__ = "0.1.0"

__all__ = [
    "AccountingCatalog", "EntityInfo",
    "ComplianceChecker", "ComplianceReport", "PostCheck", "Profile", "RuleSet",
    "Lexicon", "NormResult", "clean_name", "fold", "parse_amount", "valid_inn",
    "CachedLLM", "LLM", "OpenAICompatLLM", "QwenTransformersLLM", "ScriptedLLM",
    "ChangeType", "MemoryWriter", "WriteReport",
    "Normalizer",
    "AccountantAssistant", "AssistantReply", "ChatSession", "Extractor",
    "ReadPath", "ReadResult", "SectionClassifier", "WritePath", "WriteResult",
    "classify_query_rules",
    "ContextBuilder", "ContextBundle", "DocumentRecord",
    "QueryClassification", "QueryType", "parse_extraction",
    "POLICIES", "FactKind", "Section", "SectionPolicy",
    "EventRecord", "Fact", "MemoryStore", "RejectionRecord", "SectionState",
    "__version__",
]
