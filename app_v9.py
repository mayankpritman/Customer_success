# app_v9.py
import base64
import json
import re
import uuid
from typing import List, Dict
import pandas as pd
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
# --- your modules (already in your repo) ---
from agent import executor                      # AgentExecutor with MessagesPlaceholder("chat_history")
from eval_langfuse import run_agent_and_log     # runs agent, logs to Langfuse, returns {answer, steps, scores, trace_id}
from db_tool import execute_query               # SQLite helper
from schema_util import get_schema_description  # pretty schema text
# ------------------ page + styles ------------------
st.set_page_config(page_title="Customer Success Assistant", layout="wide")
st.markdown("""
<style>
.stApp { background: #ffffff; }
.main .block-container { max-width: 980px; }
h1,h2,h3 { color:#111827; }
hr { border-top: 1px solid #efeff4; }
.banner {
   background: linear-gradient(90deg, #6ea8ff, #8a7cff);
   padding: 14px 18px; color: white; border-radius: 14px;
   margin-top: 12px; margin-bottom: 10px;
}
.banner h2 { margin: 0; font-weight: 700; }
.banner p { margin: 4px 0 0 0; opacity: 0.95; }
.topbar { margin-top: 6px; margin-bottom: 6px; }
.try-chip {
   display:inline-block; padding:8px 12px; margin:4px 6px 4px 0; border-radius:20px;
   background:#1f6feb; color:white; font-weight:600; cursor:pointer; text-decoration:none;
}
.try-chip:hover { opacity: .92; }
.chat-assistant, .chat-user { border-radius: 12px; padding: 10px 12px; margin: 6px 0; }
.chat-user { background:#eef6ff; color:#0f172a; border:1px solid #d9e8ff; }
.chat-assistant { background:#f7f7f9; color:#111827; border:1px solid #ececf1; }
.small { opacity:.8; font-size: 12px; }
</style>
""", unsafe_allow_html=True)
# ------------------ helpers ------------------
DB_PATH = "customer_success.db"
def new_chat():
   st.session_state["chat"] = []  # [{"role":"user"|"assistant","content":str,"charts":[b64], "tables":[list[dict]], "scores":dict}]
   st.session_state["chat_id"] = str(uuid.uuid4())[:8]
   st.session_state.pop("pending_query", None)
def lc_history():
   msgs: List = []
   for m in st.session_state.get("chat", []):
       msgs.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
   return msgs
import hashlib
import json
import re
def _fingerprint_table(rows, limit=50):
   """Stable hash to detect duplicate tables coming from different tools."""
   try:
       payload = json.dumps(rows[:limit], sort_keys=True, default=str)
   except Exception:
       payload = str(rows[:limit])
   return hashlib.md5(payload.encode("utf-8")).hexdigest()
def extract_charts_and_tables(intermediate_steps, final_text):
   charts = []
   table_candidates = []
   for action, obs in intermediate_steps or []:
       # charts (from your ChartTool)
       if isinstance(obs, dict):
           if obs.get("type") == "chart" and obs.get("data"):
               charts.append(obs["data"])
           # summary tool raw_data (same rows often returned by SQL tool)
           if "raw_data" in obs and isinstance(obs["raw_data"], list) and obs["raw_data"]:
               table_candidates.append(obs["raw_data"])
       # sql tool rows: list[dict]
       if isinstance(obs, list) and obs and isinstance(obs[0], dict):
           table_candidates.append(obs)
       # also handle nested lists that may contain chart dicts
       if isinstance(obs, list):
           for it in obs:
               if isinstance(it, dict) and it.get("type") == "chart" and it.get("data"):
                   charts.append(it["data"])
   # any base64 images that the model inlined in the final text
   charts += re.findall(r"data:image/(?:png|jpeg);base64,([A-Za-z0-9+/=]+)", final_text or "")
   # ---- de-duplicate tables by fingerprint ----
   seen = set()
   unique_tables = []
   for rows in table_candidates:
       fp = _fingerprint_table(rows)
       if fp not in seen:
           seen.add(fp)
           unique_tables.append(rows)
   return charts, unique_tables

import re
def extract_sql_queries_from_steps(intermediate_steps):
   """Pull SQL SELECT queries out of LangChain intermediate_steps."""
   queries = []
   if not intermediate_steps:
       return queries
   for step in intermediate_steps:
       # Typical shape: (AgentActionMessageLog, observation)
       try:
           action, obs = step
       except Exception:
           action, obs = step, None
       candidates = []
       # 1) Whole action object as string (contains tool_input, etc.)
       candidates.append(str(action))
       # 2) tool_input field
       if hasattr(action, "tool_input"):
           candidates.append(str(action.tool_input))
       # 3) log field (the "Invoking: `sql_query` with `SELECT ...`" line)
       if hasattr(action, "log"):
           candidates.append(str(action.log))
       # 4) message_log / function_call arguments (often has the SELECT again)
       if hasattr(action, "message_log"):
           try:
               for chunk in action.message_log:
                   candidates.append(str(chunk))
           except Exception:
               pass
       # 5) Also scan the observation side (the rows)
       if isinstance(obs, dict):
           candidates.extend(map(str, obs.values()))
       elif isinstance(obs, list):
           candidates.append(str(obs))
       elif obs is not None:
           candidates.append(str(obs))
       # --- extract SELECT ... up to ; or newline ---
       for text in candidates:
           for match in re.findall(r"(?is)select\s+.*?(?:;|\n|$)", text):
               q = match.strip("` \n\t")
               if q.lower().startswith("select") and q not in queries:
                   queries.append(q)
   return queries
def list_tables() -> List[str]:
   try:
       df = execute_query(
           "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
       )
       return df["name"].tolist() if not df.empty else []
   except Exception:
       return []
# ------------------ session init ------------------
if "chat" not in st.session_state:
   new_chat()
if "variant" not in st.session_state:
   st.session_state["variant"] = "strict"  # or "baseline"
# ------------------ tabs ------------------
# Banner
   st.markdown("""
<div class="banner">
<h2>Customer Success Assistant</h2>
<p>Ask questions, analyze data, and evaluate responses.</p>
</div>
   """, unsafe_allow_html=True)
tab_chat, tab_data, tab_eval = st.tabs(["💬 Chat", "📚 Data Description", "🧪 Evaluate"])
# ================== CHAT (chat-style) ==================
with tab_chat:
   # Toolbar (right-aligned New Chat; variant optional)
   c1, c2, c3 = st.columns([2, 3, 2])
   #with c2:
   #    st.session_state["variant"] = st.selectbox(
   #        "Prompt style", ["baseline", "strict"],
   #        index=1 if st.session_state["variant"] == "strict" else 0
   #    )
   with c1:
       if st.button("➕ New Chat", use_container_width=False):
           new_chat()
           st.rerun()
   #st.markdown("---")
   # Suggestions row (chips)
   st.subheader("Assistant")
   st.caption("Try one:")
   suggested = [
       "Summarize recent tickets",
       "NPS trend last 3 months",
       "Plot ticket volume Q2",
       "Top 5 churn risks",
   ]
   chip_cols = st.columns([1,1,1,1])
   for i, text in enumerate(suggested):
       with chip_cols[i]:
           if st.button(text, key=f"sugg_{i}", use_container_width=True):
               st.session_state["pending_query"] = text
               st.rerun()
   # Render chat history
   for m in st.session_state["chat"]:
       with st.chat_message("user" if m["role"] == "user" else "assistant"):
           st.markdown(m["content"])
           # charts
           for b64 in m.get("charts", []):
               try: st.image(base64.b64decode(b64), width=720)
               except Exception: pass
           # tables
           sql_list = m.get("sql", [])  # list of SQL strings for this message
           for idx, t in enumerate(m.get("tables", [])):
               # Always show an expander above each table
               with st.expander(f"🧾 View SQL for Table {idx + 1}", expanded=False):
                   if idx < len(sql_list) and sql_list[idx]:
                       st.code(sql_list[idx], language="sql")
                   else:
                       st.code("-- SQL not captured for this table --", language="sql")
               try:
                   st.dataframe(t, use_container_width=True)
               except Exception:
                   pass
           # scores
           if m.get("scores") and m["role"] == "assistant":
               with st.expander("Scores", expanded=False):
                   st.json(m["scores"])
   # Handle pending query from chips
   pending = st.session_state.pop("pending_query", None)
   # Chat input pinned at bottom
   user_q = st.chat_input("Type your question and press Enter…")
   if user_q or pending:
       q = pending or user_q
       # show user bubble now
       st.session_state["chat"].append({"role": "user", "content": q})
       # run agent
       with st.chat_message("assistant"):
           with st.spinner("Thinking..."):
               out: Dict = run_agent_and_log(
                   executor,
                   question=q,
                   reference="",
                   variant=st.session_state["variant"],
                   chat_history=lc_history(),
               )
               answer = out.get("answer", "")
               # 1) get intermediate steps
               intermediate_steps = out.get("intermediate_steps", [])
               print("mayank intermediate_steps:", intermediate_steps)
               # 2) charts & tables
               charts, tables = extract_charts_and_tables(intermediate_steps, answer)
               # 3) SQL queries
               sql_queries = extract_sql_queries_from_steps(intermediate_steps)
               print("SQL QUERIES FOUND:", sql_queries)
               # 4) show answer
               st.markdown(answer)
               # 5) show charts
               for b64 in charts:
                   try:
                       st.image(base64.b64decode(b64), width=720)
                   except Exception:
                       pass
               # 6) show SQL + tables for THIS turn
               for idx, t in enumerate(tables):
                   with st.expander(f"🧾 View SQL for Table {idx+1}", expanded=False):
                       if idx < len(sql_queries) and sql_queries[idx]:
                           st.code(sql_queries[idx], language="sql")
                       else:
                           st.code("-- SQL not captured for this table --", language="sql")
                   st.dataframe(t, use_container_width=True)
               # 7) save in chat history
               st.session_state["chat"].append({
                   "role": "assistant",
                   "content": answer,
                   "charts": charts,
                   "tables": tables,
                   "sql": sql_queries,          # 👈 critical
                   "scores": out.get("scores"),
                   "trace_id": out.get("trace_id"),
               })
       st.rerun()  # clear chat_input and render updated history
# ================== DATA DESCRIPTION ==================
with tab_data:
   st.header("Database Schema")
   st.markdown(get_schema_description(DB_PATH))
   if st.checkbox("Show 5 rows from each table"):
       names = list_tables()
       if not names:
           st.info("No tables found or unable to read the database.")
       for name in names:
           st.subheader(name)
           try:
               st.dataframe(execute_query(f"SELECT * FROM {name} LIMIT 5"))
           except Exception as e:
               st.warning(f"Preview failed for '{name}': {e}")
# ================== EVALUATE (A/B) ==================
with tab_eval:
   st.header("Run A/B Evaluation (Baseline vs Strict; logs to Langfuse)")
   default_examples = [
       {"inputs": {"question": "Show last 5 customer interactions for ClientConnect"}, "outputs": {"answer": ""}},
       {"inputs": {"question": "Summarize NPS trend for the past 3 months for ClientConnect."}, "outputs": {"answer": ""}},
       {"inputs": {"question": "Plot ticket volume by month for Q2 for ClientConnect."}, "outputs": {"answer": ""}}
   ]
   examples_json = st.text_area("Examples JSON", value=json.dumps(default_examples, indent=2), height=220)
   if st.button("Run A/B"):
       try:
           examples = json.loads(examples_json)
       except Exception as e:
           st.error(f"Bad JSON: {e}")
       else:
           rows = []
           with st.spinner("Evaluating Baseline…"):
               for ex in examples:
                   qx = ex["inputs"]["question"]; ref = (ex.get("outputs") or {}).get("answer", "")
                   r = run_agent_and_log(executor, question=qx, reference=ref, variant="baseline", chat_history=[])
                   rows.append({"variant":"baseline", "question":qx, **(r.get("scores") or {}), "trace":r.get("trace_id")})
           with st.spinner("Evaluating Strict…"):
               for ex in examples:
                   qx = ex["inputs"]["question"]; ref = (ex.get("outputs") or {}).get("answer", "")
                   r = run_agent_and_log(executor, question=qx, reference=ref, variant="strict", chat_history=[])
                   rows.append({"variant":"strict", "question":qx, **(r.get("scores") or {}), "trace":r.get("trace_id")})
           df = pd.DataFrame(rows)
           st.success("Experiments finished. Open Langfuse (project → traces) to inspect details.")
           st.dataframe(df, use_container_width=True)
           # show means
           for c in ["correctness","conciseness","schema_guard","sql_validity"]:
               if c not in df.columns: df[c] = None
           agg = df.groupby("variant")[["correctness","conciseness","schema_guard","sql_validity"]].mean(numeric_only=True).round(3)
           st.markdown("### Mean Scores")
           st.table(agg)
