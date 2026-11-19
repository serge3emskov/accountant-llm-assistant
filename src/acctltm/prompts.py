"""Промпты LLM-шагов ассистента бухгалтера.

Принцип: extractor получает правила только выбранного раздела.
Few-shot примеры хранятся как Python-структуры и сериализуются в JSON,
поэтому они всегда синтаксически корректны; тесты дополнительно
прогоняют их через ту же валидацию, что и ответы модели.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .llm import Messages
from .schemas import QueryType
from .sections import POLICIES, Section

ROLE_LABELS = {"user": "Бухгалтер", "assistant": "Ассистент"}


def format_dialog(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{ROLE_LABELS.get(t['role'], t['role'])}: {t['content'].strip()}" for t in turns)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_BULLET = re.compile(r"^[\s\-—–*•·#>\d.)]+")


def compact_dialog(turns: list[dict[str, str]], max_questions: int = 2, max_chars: int = 300) -> list[dict[str, str]]:
    """Диалог для write-path: реплики бухгалтера целиком, от ассистента — только его последние вопросы.

    Длинные ответы ассистента не несут сведений о компании, зато в них есть названия операций
    и суммы, которые extractor может ошибочно приписать компании. Вопросы нужны как контекст:
    «нет» в ответ на «есть ли договоры?».
    """
    out: list[dict[str, str]] = []
    for t in turns:
        if t["role"] != "assistant":
            out.append({"role": t["role"], "content": t["content"]})
            continue
        sentences = [_BULLET.sub("", x).replace("**", "").strip() for x in _SENTENCE_SPLIT.split(t["content"])]
        questions = [x for x in sentences if x.endswith("?")]
        if questions:
            out.append({"role": "assistant", "content": " ".join(questions[-max_questions:])[:max_chars]})
    return out


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _fewshot(system: str, examples: list[tuple[str, dict]], query: str) -> Messages:
    msgs: Messages = [{"role": "system", "content": system}]
    for user, answer in examples:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": _j(answer)})
    msgs.append({"role": "user", "content": query})
    return msgs


# =========================================================================== 1. классификатор разделов

SECTION_DESCRIPTIONS: dict[Section, str] = {
    Section.COMPANY_PROFILE: "режим налогообложения (УСН, ОСНО, ПСН, НПД, АУСН), плательщик ли НДС, объект УСН, "
                             "численность сотрудников, выручка, освобождение от НДС",
    Section.COUNTERPARTIES: "контрагенты: с кем работаем, покупатели и поставщики, ИНН, роль; также «контрагентов нет»",
    Section.CONTRACTS: "договоры: предмет, сумма, условия оплаты (аванс/постоплата), статус; также «договоров нет»",
    Section.DOCUMENTS: "первичные документы и счета-фактуры: ожидаем, получен, проведён, отсутствует",
    Section.OBLIGATIONS: "отчётность и платежи: предстоит сдать, сдано, просрочено; также «отчётности нет»",
    Section.ISSUES: "проблемы и расхождения: не сходится касса, нет закрывающих документов, расхождение по НДС; "
                    "также «проблема устранена»",
    Section.PRACTICES: "учётные практики: наличные расчёты, онлайн-касса (ККТ), электронный документооборот (ЭДО)",
}

CLASSIFIER_SYSTEM = """Ты — модуль разметки диалогов бухгалтерского чата. Определи, в каких разделах памяти бухгалтер сообщил сведения о компании (или о другой компании группы).

Разделы:
{sections}

Правила:
- Учитывай только то, что сообщил бухгалтер. Реплики ассистента — контекст: ответ «нет» на вопрос ассистента о договорах относится к contracts.
- Вопрос бухгалтера не является сведением о компании. «Можно ли принять НДС к вычету?» не значит, что компания принимает НДС к вычету.
- Если сведений нет, верни пустой список.

Ответ — только JSON без пояснений: {{"sections": [...]}}"""

CLASSIFIER_EXAMPLES: list[tuple[list[dict[str, str]], list[str]]] = [
    (
        [{"role": "user", "content": "Мы на УСН доходы, работаем с Ромашкой как поставщик. Можно принять расходы?"}],
        ["company_profile", "counterparties"],
    ),
    (
        [
            {"role": "user", "content": "Нужно сдать отчётность."},
            {"role": "assistant", "content": "Подскажу. Есть ли действующие договоры и какие отчёты предстоят?"},
            {"role": "user", "content": "Договоров нет. Декларацию УСН ещё не сдавали."},
        ],
        ["contracts", "obligations"],
    ),
    ([{"role": "user", "content": "До скольки вы сегодня работаете?"}], []),
]


def section_classifier_messages(turns: list[dict[str, str]]) -> Messages:
    sections = "\n".join(f"- {s.value} — {d}" for s, d in SECTION_DESCRIPTIONS.items())
    system = CLASSIFIER_SYSTEM.format(sections=sections)
    examples = [(f"Диалог:\n{format_dialog(t)}", {"sections": s}) for t, s in CLASSIFIER_EXAMPLES]
    return _fewshot(system, examples, f"Диалог:\n{format_dialog(turns)}")


# =========================================================================== 2. extractor по разделу

EXTRACTOR_SYSTEM = """Ты — модуль извлечения фактов из диалога бухгалтерского чата. Извлеки сведения ТОЛЬКО для раздела «{title}» (section="{section}"). Сведения других разделов игнорируй.

Общие правила:
- Извлекай только то, что сообщил бухгалтер. Не додумывай и не добавляй знания о налогах и операциях.
- Реплики ассистента — контекст: «нет» в ответ на вопрос ассистента — это ответ бухгалтера.
- Вопрос бухгалтера («можно ли принять X?») — не факт.
- subject — о какой компании: "self", если о своей; иначе как назвал бухгалтер: "дочка", "ип директора", "филиал".
- evidence — дословная цитата из реплики бухгалтера, на которой основан факт.
- Каждый факт — отдельный элемент списка. Извлеки все факты раздела, а не только первый.
- Если бухгалтер исправляет сказанное раньше («ошибся», «на самом деле нет»), это тоже факт.
- Поля, о которых бухгалтер не сказал, не включай. Других полей не добавляй.

Правила раздела:
{rules}

Ответ — только JSON без пояснений: {output_format}
Если сведений нет: {empty}"""

_FORMAT_WITH_STATUSES = ('{"facts": [...], "section_statuses": [...]}', '{"facts": [], "section_statuses": []}')
_FORMAT_FACTS_ONLY = ('{"facts": [...]}', '{"facts": []}')

_NAME_RULE = ("- name — в начальной форме, но именно то, что назвал бухгалтер: название контрагента не заменяй "
              "на ИНН, опечатки не исправляй.")
_POLARITY_RULE = ("- polarity: \"negative\", если бухгалтер говорит, что этого нет или компания это не делает; "
                  "иначе \"positive\".")

SECTION_RULES: dict[Section, str] = {
    Section.COMPANY_PROFILE: "\n".join([
        "- Поля: section, subject, attribute, value, evidence. Поля name и polarity здесь нет.",
        "- attribute: \"regime\" (УСН/ОСНО/ПСН/НПД/АУСН/ЕСХН), \"vat_payer\" (да/нет), "
        "\"usn_object\" (доходы/доходы-расходы), \"employees\" (число), \"revenue_year\" (сумма за год), "
        "\"vat_exempt\" (да/нет — освобождение от НДС). Каждый атрибут — отдельный факт.",
        "- «упрощёнка» → УСН, «патент» → ПСН, «самозанятость» → НПД, «общий режим» → ОСНО.",
        "- «6 процентов» → usn_object доходы, «15 процентов» → доходы-расходы.",
    ]),
    Section.COUNTERPARTIES: "\n".join([
        "- Поля: section, subject, name, polarity, status, role, inn, terms, evidence.",
        _NAME_RULE,
        "- status: \"active\" — работаем; \"paused\" — приостановили; \"terminated\" — прекратили работать.",
        "- role: \"покупатель\", \"поставщик\", \"подрядчик\", \"агент\", \"прочее\" — если сказано.",
        "- inn — ИНН, если назван (только цифры). terms — условия, если сказаны.",
        "- «Прекратили / расторгли / больше не работаем с X» → status \"terminated\". "
        "«Никогда не работали с X» → polarity \"negative\".",
        "- «Контрагентов нет» → section_statuses: \"confirmed_absent\"; «не знаю, с кем работаем» → \"dont_know\".",
    ]),
    Section.CONTRACTS: "\n".join([
        "- Поля: section, subject, name, polarity, status, amount, subject, payment, number, evidence.",
        "- name — договор как назвал бухгалтер («договор поставки с Ромашкой»).",
        "- status: \"planned\" — подписан, не начат; \"active\" — действует; \"paused\"; \"terminated\" — расторгли.",
        "- amount — сумма в рублях («150 тысяч» → 150000). payment: \"аванс\", \"постоплата\", \"смешанная\".",
        "- number — номер договора, если назван. subject — предмет, если сказан.",
        "- «Договоров нет» → section_statuses: \"confirmed_absent\".",
    ]),
    Section.DOCUMENTS: "\n".join([
        "- Поля: section, subject, name, polarity, status, amount, number, counterparty, evidence.",
        "- name — документ («счёт-фактура от Ромашки», «акт выполненных работ», «накладная»).",
        "- status: \"expected\" — ожидаем; \"received\" — получен, не проведён; \"posted\" — проведён.",
        "- «Документа нет / не получили» → polarity \"negative\".",
        "- counterparty — от кого документ, если сказано. amount, number — если названы.",
    ]),
    Section.OBLIGATIONS: "\n".join([
        "- Поля: section, subject, name, polarity, status, deadline, period, evidence.",
        "- name — отчёт или платёж («декларация НДС», «6-НДФЛ», «РСВ», «уведомление ЕНП»).",
        "- status: \"upcoming\" — предстоит; \"filed\" — сдано/уплачено; \"overdue\" — просрочено.",
        "- deadline — срок, period — период, если названы.",
        "- «Отчётности нет» (например, ИП без работников) → section_statuses: \"confirmed_absent\".",
    ]),
    Section.ISSUES: "\n".join([
        "- Поля: section, subject, name, polarity, status, details, evidence.",
        "- name — проблема своими словами бухгалтера («не сходится касса», «нет закрывающих документов»).",
        "- status: \"active\" — есть сейчас; \"resolved\" — устранена.",
        "- details — подробности, если сказаны. «Проблемы нет» → polarity \"negative\".",
        "- Контрагенты и договоры сюда не относятся — только расхождения и проблемы.",
    ]),
    Section.PRACTICES: "\n".join([
        "- Поля: section, subject, attribute, value, details, evidence. Поля name и polarity здесь нет.",
        "- attribute: \"cash_settlements\" (наличные расчёты), \"kkt\" (онлайн-касса), \"edo\" (ЭДО).",
        "- value: \"да\", \"нет\" или \"иногда\". details — подробности, если сказаны.",
    ]),
}


def _fact(section: str, **kw: Any) -> dict:
    return {"section": section, **kw}


def _st(section: str, status: str, evidence: str, subject: str = "self") -> dict:
    return {"section": section, "subject": subject, "status": status, "evidence": evidence}


EXTRACTOR_EXAMPLES: dict[Section, list[tuple[list[dict[str, str]], dict]]] = {
    Section.COMPANY_PROFILE: [
        (
            [{"role": "user", "content": "Мы на УСН доходы, НДС не платим, сотрудников пять."}],
            {"facts": [
                _fact("company_profile", attribute="regime", value="УСН", evidence="Мы на УСН"),
                _fact("company_profile", attribute="usn_object", value="доходы", evidence="УСН доходы"),
                _fact("company_profile", attribute="vat_payer", value="нет", evidence="НДС не платим"),
                _fact("company_profile", attribute="employees", value=5, evidence="сотрудников пять"),
            ]},
        ),
        (
            [{"role": "user", "content": "Выручка за год 45 миллионов, режим общий."}],
            {"facts": [
                _fact("company_profile", attribute="revenue_year", value=45000000, evidence="Выручка за год 45 миллионов"),
                _fact("company_profile", attribute="regime", value="ОСНО", evidence="режим общий"),
            ]},
        ),
    ],
    Section.COUNTERPARTIES: [
        (
            [{"role": "user", "content": "Работаем с Ромашкой, они поставщик. С Васильком больше не работаем."}],
            {"facts": [
                _fact("counterparties", name="Ромашка", polarity="positive", status="active", role="поставщик",
                      evidence="Работаем с Ромашкой, они поставщик"),
                _fact("counterparties", name="Василёк", polarity="negative",
                      evidence="С Васильком больше не работаем"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Контрагентов пока нет, только начинаем."}],
            {"facts": [], "section_statuses": [
                _st("counterparties", "confirmed_absent", "Контрагентов пока нет")]},
        ),
    ],
    Section.CONTRACTS: [
        (
            [{"role": "user", "content": "Подписали договор поставки с Ромашкой на 150 тысяч, аванс 100%."}],
            {"facts": [
                _fact("contracts", name="договор поставки с Ромашкой", polarity="positive", status="active",
                      amount=150000, payment="аванс", evidence="договор поставки с Ромашкой на 150 тысяч, аванс 100%"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Договоров у нас нет."}],
            {"facts": [], "section_statuses": [_st("contracts", "confirmed_absent", "Договоров у нас нет")]},
        ),
    ],
    Section.DOCUMENTS: [
        (
            [{"role": "user", "content": "Ждём счёт-фактуру от Ромашки за март, акт уже получили."}],
            {"facts": [
                _fact("documents", name="счёт-фактура от Ромашки за март", polarity="positive", status="expected",
                      counterparty="Ромашка", evidence="Ждём счёт-фактуру от Ромашки за март"),
                _fact("documents", name="акт выполненных работ", polarity="positive", status="received",
                      evidence="акт уже получили"),
            ]},
        ),
        (
            [{"role": "user", "content": "Накладной от Василька нет, не прислали."}],
            {"facts": [
                _fact("documents", name="накладная от Василька", polarity="negative",
                      counterparty="Василёк", evidence="Накладной от Василька нет"),
            ]},
        ),
    ],
    Section.OBLIGATIONS: [
        (
            [{"role": "user", "content": "Декларацию по НДС за первый квартал сдали, а РСВ ещё предстоит."}],
            {"facts": [
                _fact("obligations", name="декларация НДС", polarity="positive", status="filed",
                      period="первый квартал", evidence="Декларацию по НДС за первый квартал сдали"),
                _fact("obligations", name="РСВ", polarity="positive", status="upcoming",
                      evidence="РСВ ещё предстоит"),
            ], "section_statuses": []},
        ),
        (
            [{"role": "user", "content": "Отчётности нет, я ИП без работников."}],
            {"facts": [], "section_statuses": [
                _st("obligations", "confirmed_absent", "Отчётности нет, я ИП без работников")]},
        ),
    ],
    Section.ISSUES: [
        (
            [{"role": "user", "content": "Не сходится касса на конец дня, расхождение по НДС разобрали."}],
            {"facts": [
                _fact("issues", name="не сходится касса", polarity="positive", status="active",
                      evidence="Не сходится касса на конец дня"),
                _fact("issues", name="расхождение по НДС", polarity="positive", status="resolved",
                      evidence="расхождение по НДС разобрали"),
            ]},
        ),
    ],
    Section.PRACTICES: [
        (
            [{"role": "user", "content": "Расчёты наличными, кассы нет, ЭДО используем."}],
            {"facts": [
                _fact("practices", attribute="cash_settlements", value="да", evidence="Расчёты наличными"),
                _fact("practices", attribute="kkt", value="нет", evidence="кассы нет"),
                _fact("practices", attribute="edo", value="да", evidence="ЭДО используем"),
            ]},
        ),
    ],
}


def extractor_messages(section: Section, turns: list[dict[str, str]]) -> Messages:
    policy = POLICIES[section]
    fmt, empty = _FORMAT_WITH_STATUSES if policy.absence_aware else _FORMAT_FACTS_ONLY
    system = EXTRACTOR_SYSTEM.format(title=policy.title, section=section.value, rules=SECTION_RULES[section],
                                     output_format=fmt, empty=empty)
    examples = [(f"Диалог:\n{format_dialog(t)}", out) for t, out in EXTRACTOR_EXAMPLES[section]]
    return _fewshot(system, examples, f"Диалог:\n{format_dialog(turns)}")


# =========================================================================== 3. классификатор запроса

QUERY_DESCRIPTIONS: dict[QueryType, str] = {
    QueryType.POSTING: "как отразить операцию, какие проводки и счета учёта использовать",
    QueryType.TAX: "налоги, взносы, ставки, как рассчитать налог",
    QueryType.DOCUMENT_CHECK: "проверка документа, контрагента, счёта-фактуры, можно ли доверять контрагенту",
    QueryType.DEDUCTION: "можно ли принять расход к учёту или НДС к вычету",
    QueryType.DEADLINE: "сроки сдачи отчётности и уплаты налогов",
    QueryType.HISTORY: "что было раньше: договоры, контрагенты, сданная отчётность",
    QueryType.PROFILE_REVIEW: "что ассистент знает о компании",
    QueryType.GENERAL: "не связано с учётом компании (общие вопросы, приветствие), а также сообщение без вопроса, "
                       "где бухгалтер только рассказывает о компании",
}

QUERY_SYSTEM = """Ты — классификатор сообщений бухгалтера в чате. Определи тип последнего сообщения и о какой компании оно.

Типы (можно несколько):
{types}

subject — о какой компании: "self", если о своей или не указано; иначе как назвал бухгалтер: "дочка", "ип директора", "филиал".

Ответ — только JSON без пояснений: {{"query_types": [...], "subject": "..."}}"""

QUERY_EXAMPLES: list[tuple[str, dict]] = [
    ("Как провести покупку товаров у Ромашки, какие проводки?",
     {"query_types": ["posting"], "subject": "self"}),
    ("Можно принять НДС к вычету без счёта-фактуры?",
     {"query_types": ["deduction", "document_check"], "subject": "self"}),
    ("Когда сдавать декларацию по УСН?", {"query_types": ["deadline"], "subject": "self"}),
    ("Какая ставка НДС в этом году?", {"query_types": ["tax"], "subject": "self"}),
    ("Что ты знаешь о нашей компании?", {"query_types": ["profile_review"], "subject": "self"}),
    ("Здравствуйте! Мы на упрощёнке.", {"query_types": ["general"], "subject": "self"}),
]


def query_classifier_messages(message: str, history: list[dict[str, str]] | None = None) -> Messages:
    types = "\n".join(f"- {q.value} — {d}" for q, d in QUERY_DESCRIPTIONS.items())
    system = QUERY_SYSTEM.format(types=types)
    examples = [(f"Сообщение бухгалтера: {m}", out) for m, out in QUERY_EXAMPLES]
    ctx = ""
    if history:
        ctx = f"Предыдущие реплики:\n{format_dialog(history[-4:])}\n\n"
    return _fewshot(system, examples, f"{ctx}Сообщение бухгалтера: {message}")


# =========================================================================== 4. ответ ассистента

ANSWER_SYSTEM = """Ты — ассистент бухгалтера в онлайн-чате. Отвечай по-русски, коротко: не больше 6 предложений, без заголовков и эмодзи.

Правила:
- Ты помощник, а не главный бухгалтер и не аудитор. Не даёшь юридических гарантий; при сложных или спорных операциях советуй свериться с главбухом или действующей редакцией НК РФ.
- Ставки, лимиты и нормы бери только из блока СПРАВОЧНИК и помечай, что их нужно проверить на актуальность. Не выдумывай точные суммы налогов и штрафов.
- Учитывай блок ПАМЯТЬ: режим, НДС, объект УСН, численность, выручку, договоры, контрагентов, практики. Если что-то из этого влияет на ответ, прямо скажи об этом («учитываю, что вы на УСН доходы…»).
- Пункт с пометкой НЕТ означает, что этого у компании нет.
- Если есть блок «НУЖНО УТОЧНИТЬ», сначала задай эти вопросы — кроме тех, на которые бухгалтер уже ответил в этом диалоге.
- Если бухгалтер только рассказал о компании и ничего не спросил, коротко подтверди и спроси, чем помочь.
- Если сведения из памяти и из документированного источника (выписки) расходятся, скажи об этом и уточни.
- Не придумывай сведения о компании и контрагентах.
- При признаках риска (контрагент с риск-флагами, наличные сверх лимита, расход без документов) прямо назови риск.

{context}"""


def answer_messages(context: str, history: list[dict[str, str]], message: str) -> Messages:
    msgs: Messages = [{"role": "system", "content": ANSWER_SYSTEM.format(context=context)}]
    msgs += [{"role": t["role"], "content": t["content"]} for t in history]
    msgs.append({"role": "user", "content": message})
    return msgs
