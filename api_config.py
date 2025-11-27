# api_config.py
import json
import os
from pathlib import Path
try:
   import streamlit as st  # available on Streamlit Cloud
except ModuleNotFoundError:
   st = None
CONFIG_PATH = Path(__file__).with_name("api_config.json")
DEFAULTS = {
   "api_choice": "azure",
   "azure": {
       "api_key": "",
       "endpoint": "",
       "deployment": "",
       "api_version": "2024-02-01",
   },
   "openai": {"api_key": "", "model": "gpt-4o-mini"},
   "langfuse": {
       "enabled": False,
       "public_key": "",
       "secret_key": "",
       "host": "https://cloud.langfuse.com",
       "project": "CustomerSuccessAssistant",
   },
}

def _get_secret(name: str, default: str | None = None) -> str | None:
   """Read from env first, then from st.secrets (Streamlit Cloud)."""
   if os.getenv(name):
       return os.getenv(name)
   if st is not None and name in st.secrets:
       return st.secrets[name]
   return default

def load_api_details(path: str | Path = CONFIG_PATH) -> dict:
   """Load config from JSON (if present) and merge with defaults + secrets."""
   cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
   path = Path(path)
   if path.exists():
       try:
           with path.open(mode="r", encoding="utf-8") as f:
               file_cfg = json.load(f)
           # shallow merge dicts
           for k, v in file_cfg.items():
               if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                   cfg[k].update(v)
               else:
                   cfg[k] = v
       except json.JSONDecodeError as e:
           raise RuntimeError(
               f"Invalid JSON in {path}: {e.msg} at line {e.lineno}, col {e.colno}"
           )
   # ---- inject secrets ----
   # Azure/OpenAI
   az = cfg.setdefault("azure", {})
   az_key = az.get("api_key") or _get_secret("AZURE_OPENAI_API_KEY") or _get_secret(
       "OPENAI_API_KEY"
   )
   if az_key:
       az["api_key"] = az_key
   op = cfg.setdefault("openai", {})
   op_key = op.get("api_key") or _get_secret("OPENAI_API_KEY")
   if op_key:
       op["api_key"] = op_key
   # Langfuse
   lf = cfg.setdefault("langfuse", {})
   if lf.get("enabled"):
       lf_pk = lf.get("public_key") or _get_secret("LANGFUSE_PUBLIC_KEY")
       lf_sk = lf.get("secret_key") or _get_secret("LANGFUSE_SECRET_KEY")
       lf_host = lf.get("host") or _get_secret(
           "LANGFUSE_HOST", "https://cloud.langfuse.com"
       )
       lf_proj = lf.get("project") or _get_secret("LANGFUSE_PROJECT")
       if lf_pk:
           os.environ["LANGFUSE_PUBLIC_KEY"] = lf_pk
       if lf_sk:
           os.environ["LANGFUSE_SECRET_KEY"] = lf_sk
       if lf_host:
           os.environ["LANGFUSE_HOST"] = lf_host
       if lf_proj:
           os.environ["LANGFUSE_PROJECT"] = lf_proj
   # export OpenAI key for libraries that expect it
   if op.get("api_key"):
       os.environ.setdefault("OPENAI_API_KEY", op["api_key"])
   return cfg

def build_llm_from_cfg(cfg: dict):
   from langchain.chat_models import AzureChatOpenAI
   from langchain_openai import ChatOpenAI
   choice = cfg.get("api_choice", "azure").lower()
   if choice.startswith("azure"):
       az = cfg["azure"]
       return AzureChatOpenAI(
           api_key=az["api_key"],
           azure_endpoint=az["endpoint"],
           azure_deployment=az["deployment"],
           api_version=az.get("api_version", "2024-02-01"),
           temperature=0,
       )
   else:
       return ChatOpenAI(model=cfg["openai"]["model"], temperature=0)

# ---------- Langfuse helpers ----------
class _NoopTrace:
   def __init__(self, *_, **__): ...
   def trace(self, *_, **__): return self
   def score(self, *_, **__): return None
   def flush(self, *_, **__): return None
   def create_observation(self, *_, **__): return None
   def create_generation(self, *_, **__): return None
   def create_score(self, *_, **__): return None

def build_langfuse_client():
   """Return real Langfuse client if keys exist, otherwise a no-op."""
   from langfuse import Langfuse
   pk = _get_secret("LANGFUSE_PUBLIC_KEY")
   sk = _get_secret("LANGFUSE_SECRET_KEY")
   host = _get_secret("LANGFUSE_HOST", "https://cloud.langfuse.com")
   if pk and sk:
       return Langfuse(public_key=pk, secret_key=sk, host=host)
   return _NoopTrace()

def get_project_from_cfg(cfg: dict) -> str:
   return (cfg.get("langfuse") or {}).get("project", "CustomerSuccessAssistant")
