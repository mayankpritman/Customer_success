# agent.py

from __future__ import annotations
from typing import List, Dict, Optional, Any
# LangChain core

from langchain.agents import Tool, AgentExecutor

from langchain.chains import LLMChain

# ✅ Import MessagesPlaceholder

try:

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

except Exception:

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # LC >= 0.2

# Prefer the modern functions agent; keep fallback

try:

    from langchain.agents import create_openai_functions_agent

except Exception:

    create_openai_functions_agent = None

try:

    from langchain.agents import initialize_agent

except Exception:

    initialize_agent = None

# Your modules

from api_config import load_api_details, build_llm_from_cfg

from db_tool import execute_query, generate_chart

from schema_util import get_schema_description

# Optional synonyms

try:

    from column_synonyms import COLUMN_SYNONYMS, normalize_query_columns

    _HAS_SYNONYMS = True

except Exception:

    COLUMN_SYNONYMS = {}

    _HAS_SYNONYMS = False


    def normalize_query_columns(x: str) -> str:

        return x

DB_PATH = "customer_success.db"

cfg = load_api_details()

llm = build_llm_from_cfg(cfg)

SCHEMA_MD = get_schema_description(DB_PATH)

ALIASES_MD = ""

if _HAS_SYNONYMS and COLUMN_SYNONYMS:
    ALIASES_MD = "\n".join(f"- **{k}** → `{v}`" for k, v in COLUMN_SYNONYMS.items())

SYSTEM_PROMPT = (
        "You are a Customer Success data assistant. You can:\n"
        " • run SQL queries against the SQLite database,\n"
        "Perform case-insensitive filtering using LOWER(column_name) = 'value'\n"
        " • generate charts  from query results,\n"
        " • summarize query results.\n\n"
        "Use ONLY the tables and columns defined in the schema below. Do not invent names. "
        "If user mentions an alias, map it to the real column from the alias list.\n\n"
        "### Database Schema\n"
        f"{SCHEMA_MD}\n\n"
        + ("### Column Aliases\n" + ALIASES_MD + "\n\n" if ALIASES_MD else "")
        + "Behavioral rules:\n"
          " - Prefer precise SQL with LIMITs and exact column names.\n"
          " - If a chart is requested and axes aren’t specified, use the first two columns.\n"
          " - If both a summary and a chart are requested, call both tools.\n"

)


# ── Tools ──────────────────────────────────────────────────────────────────────

def _sql_query_fn(query: str) -> List[Dict[str, Any]]:
    real_sql = normalize_query_columns(query)

    df = execute_query(real_sql)

    return df.to_dict(orient="records")


def _chart_fn(query: str, x_col: Optional[str] = None, y_col: Optional[str] = None, chart_type: str = "bar") -> Dict[
    str, Any]:
    real_sql = normalize_query_columns(query)

    df = execute_query(real_sql)

    if df.empty:
        return {"type": "error", "message": "No data returned by the query."}

    if not x_col or not y_col:

        cols = list(df.columns)

        if len(cols) < 2:
            return {"type": "error", "message": "Need at least two columns (x_col, y_col)."}

        x_col, y_col = cols[0], cols[1]

    return generate_chart(real_sql, x_col, y_col, chart_type)


def _summarize_fn(query: str) -> Dict[str, Any]:
    real_sql = normalize_query_columns(query)

    df = execute_query(real_sql)

    if df.empty:
        return {"summary": "No data returned.", "raw_data": []}

    preview_csv = df.head(5).to_csv(index=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer success analyst. Summarize the following table, generate charts also if possible and give suggestions as well"),
        ("user", "{table_csv}")

    ])

    summary = LLMChain(llm=llm, prompt=prompt).predict(table_csv=preview_csv).strip()

    return {"summary": summary, "raw_data": df.to_dict(orient="records")}


TOOLS = [

    Tool(name="sql_query", func=_sql_query_fn, description="Run a SQL SELECT and return rows as JSON."),

    Tool(name="generate_chart", func=_chart_fn,
         description="Create a bar/line chart from a SQL query and give summary of that graph also. Args: x_col, y_col, chart_type."),

    Tool(name="summarize_query", func=_summarize_fn,
         description="Summarize a SQL SELECT result; returns summary + raw rows."),

]


# ── Agent ──────────────────────────────────────────────────────────────────────

def build_executor() -> AgentExecutor:
   if create_openai_functions_agent is None:
       raise RuntimeError("Please upgrade langchain to a version with create_openai_functions_agent")
   prompt = ChatPromptTemplate.from_messages([
       ("system", SYSTEM_PROMPT),
       MessagesPlaceholder(variable_name="chat_history"),   # ← REQUIRED
       ("human", "{input}"),
       MessagesPlaceholder(variable_name="agent_scratchpad"),
   ])
   agent_runnable = create_openai_functions_agent(llm=llm, tools=TOOLS, prompt=prompt)
   return AgentExecutor(
       agent=agent_runnable,
       tools=TOOLS,
       verbose=True,
       return_intermediate_steps=True,
       handle_parsing_errors=True,
   )
executor = build_executor()
