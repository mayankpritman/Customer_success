# eval_langfuse.py
from __future__ import annotations
import json, time, re, sqlite3
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from api_config import (
   load_api_details,
   build_llm_from_cfg,
   build_langfuse_client,
   get_project_from_cfg,
)
from schema_util import get_schema_description
# ── bootstrap config / clients ────────────────────────────────────────────────
_cfg = load_api_details()
_llm = build_llm_from_cfg(_cfg)
_lf  = build_langfuse_client()
_PROJECT = get_project_from_cfg(_cfg)
_DB_PATH = "customer_success.db"
# ── compatibility shims (support older/newer langfuse SDKs) ───────────────────
def _trace(name: str, metadata: dict, project: str):
   if hasattr(_lf, "trace"):      # new SDK
       return _lf.trace(name=name, metadata=metadata, project=project)
   return _lf.create_trace(name=name, metadata=metadata, project=project)
def _trace_id(t):
   return getattr(t, "id", None) or (t.get("id") if isinstance(t, dict) else None)
def _event(trace, **kwargs):
   if hasattr(trace, "event"):
       return trace.event(**kwargs)
   return _lf.create_event(trace_id=_trace_id(trace), **kwargs)
def _generation(trace, **kwargs):
   if hasattr(trace, "generation"):
       return trace.generation(**kwargs)
   return _lf.create_generation(trace_id=_trace_id(trace), **kwargs)
def _obs_id(obs):
   return getattr(obs, "id", None) or (obs.get("id") if isinstance(obs, dict) else None)
def _score(**kwargs):
   if hasattr(_lf, "score"):
       return _lf.score(**kwargs)
   return _lf.create_score(**kwargs)
# ── A/B prompt variants ───────────────────────────────────────────────────────
PROMPT_BASELINE = "Follow the database schema strictly. Answer succinctly.\n\nUser: {question}"
PROMPT_STRICT   = ("CRITICAL: Use ONLY columns/tables from the schema; never hallucinate. "
                  "Prefer SQL that limits rows and exact fields. Return a concise, structured answer.\n\n"
                  "If both a chart and a summary are requested, do both.\n\nUser: {question}")
# ── logging helpers ───────────────────────────────────────────────────────────
def start_trace(meta: Dict | None = None):
   return _trace(name="agent_run", metadata=meta or {}, project=_PROJECT)
def log_tool_step(trace, name: str, input_obj, output_obj):
   _event(trace, name=name, input=input_obj, output=output_obj)
def log_final_generation(trace, question: str, answer: str, model_name: str):
   gen = _generation(trace, name="final_answer", input=question, output=answer, model=model_name)
   return _obs_id(gen)
# ── LLM judges ────────────────────────────────────────────────────────────────
def _azure_judge(system_instr: str, question: str, answer: str, reference: str = "") -> Dict:
   # Escape braces in the example JSON with {{ }} because ChatPromptTemplate treats {} as variables.
   system_text = (
       system_instr
       + "\nReturn ONLY JSON with this schema:\n"
       + "{{\"score\": <float 0..1>, \"reason\": \"<short explanation>\"}}"
       + "\nDo not include code fences or extra text."
   )
   prompt = ChatPromptTemplate.from_messages([
       ("system", system_text),
       ("user",
        "User question:\n{question}\n\n"
        "Agent answer:\n{answer}\n\n"
        "Reference (may be empty):\n{reference}\n")
   ])
   text = LLMChain(llm=_llm, prompt=prompt).predict(
       question=question, answer=answer, reference=reference
   ).strip()
   try:
       obj = json.loads(text)
       score = float(obj.get("score", 0))
       reason = obj.get("reason", "")
   except Exception:
       m = re.search(r"([01](?:\.\d+)?)", text)
       score = float(m.group(1)) if m else 0.0
       reason = f"Unparseable judge output: {text[:200]}"
   score = max(0.0, min(1.0, score))
   return {"score": score, "reason": reason}
def judge_correctness(q: str, a: str, r: str = "") -> Dict:
   return _azure_judge("Judge how well the answer matches the user's intent and the reference.", q, a, r)
def judge_conciseness(q: str, a: str, r: str = "") -> Dict:
   return _azure_judge("Judge whether the answer is clear and concise while preserving necessary details.", q, a, r)
# ── Schema guard (rule-based) ─────────────────────────────────────────────────
def schema_guard(answer_text: str) -> Dict:
   schema_md = get_schema_description(_DB_PATH)
   known = set()
   for line in schema_md.splitlines():
       if line.startswith("|") and "|" in line[1:]:
           parts = [p.strip() for p in line.split("|")[1:-1]]
           if parts:
               known.add(parts[0].strip("` "))
       if line.startswith("## `") and "`" in line[4:]:
           known.add(line.split("`")[1])
   bad = []
   for tok in set(t.strip("` ,.:;()").lower() for t in answer_text.split()):
       if tok and tok.isidentifier() and tok not in (k.lower() for k in known):
           if "_" in tok or tok.endswith("_id"):
               bad.append(tok)
   score = 1.0 if not bad else max(0.0, 1.0 - 0.2 * len(bad))
   reason = "OK" if not bad else f"Unknown identifiers: {sorted(bad)[:6]}"
   return {"score": score, "reason": reason}
# ── SQL validity metric ───────────────────────────────────────────────────────
def _strip_semicolon(sql: str) -> str:
   return sql.strip().rstrip(";").strip()
def _explain_sql(db_path: str, sql: str) -> tuple[bool, str]:
   sql = _strip_semicolon(sql)
   try:
       with sqlite3.connect(db_path) as conn:
           cur = conn.cursor()
           cur.execute(f"EXPLAIN QUERY PLAN {sql}")
           plan = cur.fetchall()
       return True, f"Plan rows: {len(plan)}"
   except Exception as e:
       return False, f"{type(e).__name__}: {e}"
def _probe_sql(db_path: str, sql: str) -> tuple[bool, str]:
   sql = _strip_semicolon(sql)
   wrapped = f"SELECT * FROM ({sql}) LIMIT 1"
   try:
       with sqlite3.connect(db_path) as conn:
           cur = conn.cursor()
           cur.execute(wrapped)
           _ = cur.fetchone()
       return True, "Executed with LIMIT 1"
   except Exception as e:
       return False, f"{type(e).__name__}: {e}"
def _score_from_err(err_msg: str) -> float:
   em = err_msg.lower()
   if any(x in em for x in ["no such table", "no such column", "syntax error"]):
       return 0.0
   return 0.4  # partial credit for non-fatal/unknown errors
def validate_single_sql(sql: str, db_path: str = _DB_PATH) -> dict:
   if not sql or not sql.strip():
       return {"score": 0.0, "reason": "Empty SQL."}
   if not re.match(r"^\s*select\b", sql, re.I):
       return {"score": 0.0, "reason": "Non-SELECT SQL not allowed."}
   ok1, msg1 = _explain_sql(db_path, sql)
   if not ok1:
       return {"score": _score_from_err(msg1), "reason": f"EXPLAIN failed: {msg1}"}
   ok2, msg2 = _probe_sql(db_path, sql)
   if not ok2:
       return {"score": _score_from_err(msg2), "reason": f"Probe failed: {msg2}"}
   return {"score": 1.0, "reason": "Explain OK; probe OK."}
def _extract_sqls_from_steps(steps) -> List[str]:
   sqls = []
   for action, _obs in steps or []:
       try:
           if action.tool not in {"sql_query", "generate_chart", "summarize_query"}:
               continue
           ti = action.tool_input
           if isinstance(ti, str):
               sqls.append(ti)
           elif isinstance(ti, dict) and isinstance(ti.get("query"), str):
               sqls.append(ti["query"])
       except Exception:
           continue
   return [s for s in sqls if s and s.strip()]
def sql_validity_from_steps(steps, db_path: str = _DB_PATH) -> dict:
   sqls = _extract_sqls_from_steps(steps)
   if not sqls:
       return {"score": None, "reason": "No SQL tool executions found."}
   scores, reasons = [], []
   for s in sqls:
       r = validate_single_sql(s, db_path=db_path)
       scores.append(r["score"])
       reasons.append(r["reason"])
   final = min(scores)  # penalize worst query (change to sum/avg if you prefer)
   return {"score": final, "reason": "; ".join(sorted(set(reasons)))}
# ── attach + run helpers ──────────────────────────────────────────────────────
def attach_scores(trace, observation_id: str, question: str, answer: str, reference: str = "") -> Dict[str, float]:
   corr = judge_correctness(question, answer, reference)
   conc = judge_conciseness(question, answer, reference)
   sgd  = schema_guard(answer)
   _score(name="correctness",  value=float(corr["score"]), comment=corr["reason"],
          trace_id=_trace_id(trace), observation_id=observation_id)
   _score(name="conciseness",  value=float(conc["score"]), comment=conc["reason"],
          trace_id=_trace_id(trace), observation_id=observation_id)
   _score(name="schema_guard", value=float(sgd["score"]),  comment=sgd["reason"],
          trace_id=_trace_id(trace), observation_id=observation_id)
   if hasattr(_lf, "flush"): _lf.flush()
   return {"correctness": corr["score"], "conciseness": conc["score"], "schema_guard": sgd["score"]}
def run_agent_and_log(
   executor,
   question: str,
   reference: str = "",
   variant: str = "baseline",
   chat_history: Optional[List[Any]] = None,   # chat history is optional, but...
) -> Dict:
   tpl = PROMPT_STRICT if variant == "strict" else PROMPT_BASELINE
   q = tpl.format(question=question)
   trace = start_trace({"variant": variant})
   t0 = time.time()
   # ✅ ALWAYS pass chat_history (empty list if none)
   inputs = {"input": q, "chat_history": chat_history or []}
   result = executor.invoke(inputs)
   latency = round(time.time() - t0, 3)
   steps = result.get("intermediate_steps", [])
   for action, obs in steps:
       try:
           log_tool_step(trace, name=f"tool:{action.tool}", input_obj=action.tool_input, output_obj=obs)
       except Exception:
           pass
   model_name = (_cfg["azure"]["deployment"]
                 if _cfg["api_choice"].lower().startswith("azure")
                 else _cfg["openai"]["model"])
   gen_id = log_final_generation(trace, question=q, answer=result.get("output", ""), model_name=model_name)
   scores = attach_scores(trace, gen_id, question=q, answer=result.get("output", ""), reference=reference or "")
   sv = sql_validity_from_steps(steps, db_path=_DB_PATH)
   if sv["score"] is not None:
       _score(name="sql_validity",
              value=float(sv["score"]),
              comment=sv["reason"],
              trace_id=_trace_id(trace),
              observation_id=gen_id)
       if hasattr(_lf, "flush"): _lf.flush()
       scores["sql_validity"] = sv["score"]
   else:
       scores["sql_validity"] = None
   return {
       "trace_id": _trace_id(trace),
       "latency_s": latency,
       "answer": result.get("output", ""),
       "scores": scores,
       "intermediate_steps": steps,
   }