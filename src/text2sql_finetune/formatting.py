from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "You are a careful Text-to-SQL assistant. Generate exactly one valid SQLite SQL query "
    "for the user question.\n"
    "Rules:\n"
    "1. Use only the tables and columns that appear in the provided database schema.\n"
    "2. Do not invent table names, column names, or join keys.\n"
    "3. When using table aliases, make sure every column belongs to the table behind that alias.\n"
    "4. For JOIN conditions, prefer the provided foreign key relationships when available.\n"
    "5. Keep the SQL executable in SQLite.\n"
    "6. Return SQL only, without markdown, explanation, or comments."
)


def build_prompt(question: str, schema_context: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        "You must answer the question using only this database schema.\n"
        "The quoted table names and column names below are the only valid names.\n\n"
        "Database schema:\n"
        f"{schema_context.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "Generate the SQLite SQL query now.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_repair_prompt(question: str, schema_context: str, bad_sql: str, error: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        "The previous SQL failed when executed on SQLite. Correct it using only the schema below.\n"
        "Check table aliases, column ownership, join keys, and SQLite syntax carefully.\n\n"
        "Database schema:\n"
        f"{schema_context.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "The previous SQL failed.\n"
        f"Previous SQL:\n{bad_sql.strip()}\n\n"
        f"SQLite error:\n{error.strip()}\n\n"
        "Return one corrected executable SQLite SQL query only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_sql(text: str) -> str:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
    for stop in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    # Keep quoted semicolons, string whitespace and comments intact. SQLite
    # rejects multiple statements; never silently discard a second statement.
    return text.strip()


def normalize_sql(sql: str) -> str:
    from text2sql_finetune.sql_execution import normalize_sql as safe_normalize

    return safe_normalize(extract_sql(sql))


def is_readonly_select(sql: str) -> bool:
    # This is a quick UI check. The SQLite authorizer is the enforcement layer.
    raw = extract_sql(sql)
    unquoted = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\]", " ", raw)
    unquoted = re.sub(r"--[^\n]*|/\*.*?\*/", " ", unquoted, flags=re.DOTALL)
    normalized = unquoted.rstrip(";").strip().lower()
    if not normalized:
        return False
    blocked = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "truncate",
        "attach",
        "detach",
        "pragma",
        "vacuum",
    }
    first = normalized.split(" ", 1)[0]
    if first not in {"select", "with"}:
        return False
    if ";" in normalized:
        return False
    return not any(re.search(rf"\b{kw}\b", normalized) for kw in blocked)
