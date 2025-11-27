# eval_runner.py  — Azure-only evaluators
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple
from statistics import mean
from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from schema_util import get_schema_description
from api_config import load_api_details, build_llm_from_cfg  # uses your Azure config
# ────────────────────────────────────────────────────────────────────────────────
# Build a single Azure LLM instance for evaluators (reuses your config)
_cfg_for_eval = load_api_details()                 # reads api_config.json
_AZURE_LLM     = build_llm_from_cfg(_cfg_for_eval) # AzureChatOpenAI instance
# A/B prompt wrappers (we prepend these to the user question before calling your agent)
PROMPT_BASELINE = "Follow the database schema strictly. Answer succinctly.\n\nUser: {question}"
PROMPT_STRICT   = ("CRITICAL: Use ONLY columns/tables from the schema; never hallucinate. "
                  "Prefer SQL that limits rows and returns the exact fields asked. "
                  "Return a concise, well-structured answer.\n\nUser: {question}")

# ───────────────────────── helpers to call your agent ──────────────────────────
def _wrap_agent(executor, question: str) -> Dict:
   res = executor.invoke({"input": question})
   return {
       "answer": res.get("output", ""),
       "trajectory": str(res.get("intermediate_steps", "")),
   }
def target_prompt(variant: str, executor):
   tpl = PROMPT_STRICT if variant == "strict" else PROMPT_BASELINE
   def _fn(inputs: Dict) -> Dict:
       return _wrap_agent(executor, tpl.format(question=inputs["question"]))
   return _fn

# ─────────────────────── Azure-backed LLM-as-judge evaluators ──────────────────
def azure_llm_judge(feedback_key: str, system_instr: str):
   """
   Build a code evaluator that uses your Azure LLM to return a score in [0,1]
   and a short reason. It expects JSON: {"score": <0..1>, "reason": "..."}.
   """
   prompt = ChatPromptTemplate.from_messages([
       ("system", system_instr + "\n\n"
                  "Respond ONLY with JSON like: {\"score\": 0.0-1.0, \"reason\": \"...\"}"),
       ("user",
        "User question:\n{question}\n\n"
        "Agent answer:\n{answer}\n\n"
        "Reference answer (may be empty):\n{reference}\n")
   ])
   chain = LLMChain(llm=_AZURE_LLM, prompt=prompt)
   def _eval(run, example: Example | None = None) -> EvaluationResult:
       inputs = run.inputs or {}
       outputs = run.outputs or {}
       refs = (example.outputs if example else {}) or {}
       text = chain.predict(
           question=inputs.get("question", ""),
           answer=outputs.get("answer", ""),
           reference=refs.get("answer", ""),
       ).strip()
       # Parse {"score": ..., "reason": "..."}
       try:
           obj = json.loads(text)
           score = float(obj.get("score", 0))
           reason = obj.get("reason", "")
       except Exception:
           # fallback: try to pull a float from text
           import re
           match = re.search(r"([01](?:\.\d+)?)", text)
           score = float(match.group(1)) if match else 0.0
           reason = f"Unparseable judge output: {text[:200]}"
       score = max(0.0, min(1.0, score))
       return EvaluationResult(key=feedback_key, score=score, value=reason)
   return _eval

# ───────────────────────────── schema guard evaluator ───────────────────────────
def schema_guard_evaluator(run, example: Example | None = None) -> EvaluationResult:
   """Penalize identifiers that don't appear in DB schema (rough heuristic)."""
   schema_md = get_schema_description("customer_success.db")
   known = set()
   for line in schema_md.splitlines():
       if line.startswith("|") and "|" in line[1:]:
           parts = [p.strip() for p in line.split("|")[1:-1]]
           if parts: known.add(parts[0].strip("` "))
       if line.startswith("## `") and "`" in line[4:]:
           known.add(line.split("`")[1])
   text = (run.outputs or {}).get("answer", "") or ""
   bad = []
   for tok in set(t.strip("` ,.:;()").lower() for t in text.split()):
       if tok and tok.isidentifier() and tok not in (k.lower() for k in known):
           if "_" in tok or tok.endswith("_id"):
               bad.append(tok)
   score = 1.0 if not bad else max(0.0, 1.0 - 0.2 * len(bad))
   comment = "OK" if not bad else f"Unknown identifiers: {sorted(bad)[:6]}"
   return EvaluationResult(key="schema_guard", score=score, comment=comment)

def build_evaluators():
   """Three evaluators: correctness, conciseness (Azure), and schema guard."""
   return [
       azure_llm_judge(
           "correctness",
           "Judge how well the answer matches the user's intent and reference. "
           "Penalize missing key facts, contradictions, or fabrications."
       ),
       azure_llm_judge(
           "conciseness",
           "Judge if the answer is clear and concise while preserving necessary details."
       ),
       schema_guard_evaluator,
   ]

# ───────────────────── dataset + experiment runner (with links) ─────────────────
def _ensure_dataset(client: Client, name: str) -> str:
   try:
       return client.create_dataset(dataset_name=name, description="CSA eval set").id
   except Exception:
       for ds in client.list_datasets():
           if ds.name == name:
               return ds.id
       raise
def _upsert_examples(client: Client, dataset_id: str, examples: List[Dict]):
   client.create_examples(dataset_id=dataset_id, examples=examples)
def _app_base_from_env() -> str:
   ep = (
       os.getenv("LANGSMITH_ENDPOINT")
       or os.getenv("LANGCHAIN_ENDPOINT")
       or "https://api.smith.langchain.com"
   ).rstrip("/")
   if "api.smith.langchain.com" in ep:
       return ep.replace("api.", "")
   if "smith.langchain.com" in ep:
       return ep
   return "https://smith.langchain.com"
def _extract_experiment_id(exp_res):
   for cand in (
       getattr(getattr(exp_res, "experiment", None), "id", None),
       getattr(exp_res, "id", None),
       getattr(exp_res, "experiment_id", None),
   ):
       if cand: return cand
   for attr in ("experiments", "results", "evaluations", "runs"):
       items = getattr(exp_res, attr, None)
       if items:
           for it in items:
               cand = (
                   getattr(getattr(it, "experiment", None), "id", None)
                   or getattr(it, "id", None)
                   or getattr(it, "experiment_id", None)
               )
               if cand: return cand
   return None
def run_experiment(executor, examples: List[Dict], dataset_name: str,
                  project_name: str, variant: str) -> Tuple[str, str]:
   """Runs one experiment in LangSmith and returns (name, URL)."""
   client = Client()  # uses env set by your api_config loader
   dataset_id = _ensure_dataset(client, dataset_name)
   _upsert_examples(client, dataset_id, examples)
   exp_res = client.evaluate(
       target_prompt(variant, executor),
       data=dataset_id,
       evaluators=build_evaluators(),
       experiment_prefix=f"{project_name}-{variant}",
       max_concurrency=2,
   )
   exp_obj  = getattr(exp_res, "experiment", None)
   exp_name = getattr(exp_obj, "name", None) or getattr(exp_res, "name", "Experiment")
   exp_id   = _extract_experiment_id(exp_res)
   app_base = _app_base_from_env()
   url = f"{app_base}/experiments/{exp_id}" if exp_id else app_base
   return exp_name, url

# ─────────── optional utility to summarize means in your Streamlit tab ─────────
def summarize_experiment(exp_id: str) -> Dict[str, float | None]:
   """Compute simple means of your three metrics for a given experiment id."""
   client = Client()
   fb = list(client.list_feedback(experiment_ids=[exp_id]))
   buckets = {"correctness": [], "conciseness": [], "schema_guard": []}
   for f in fb:
       k = getattr(f, "key", None)
       s = getattr(f, "score", None)
       if k in buckets and isinstance(s, (int, float)):
           buckets[k].append(float(s))
   return {k: (round(mean(v), 3) if v else None) for k, v in buckets.items()}