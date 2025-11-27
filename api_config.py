# api_config.py
import json, os
from pathlib import Path
CONFIG_PATH = Path(__file__).with_name("api_config.json")
DEFAULTS = {
   "api_choice": "azure",
   "azure": {"api_key": "", "endpoint": "", "deployment": "", "api_version": "2024-02-01"},
   "openai": {"api_key": "", "model": "gpt-4o-mini"},
   "langfuse": {"enabled": False, "public_key": "", "secret_key": "", "host": "https://cloud.langfuse.com", "project": "CustomerSuccessAssistant"}
}
def save_api_details(api_choice: str, api_key: str, azure_endpoint=None, azure_deployment=None, api_version="2024-02-01"):
   cfg = DEFAULTS.copy()
   cfg["api_choice"] = api_choice
   if api_choice.lower().startswith("azure"):
       cfg["azure"].update({"api_key": api_key, "endpoint": azure_endpoint or "", "deployment": azure_deployment or "", "api_version": api_version})
   else:
       cfg["openai"]["api_key"] = api_key
   with CONFIG_PATH.open("w", encoding="utf-8") as f:
       json.dump(cfg, f, indent=2)
def load_api_details(path: str | Path = CONFIG_PATH) -> dict:
   path = Path(path)
   if not path.exists():
       raise FileNotFoundError(f"{path} not found")
   try:
       with path.open("r", encoding="utf-8") as f:
           cfg = json.load(f)
   except json.JSONDecodeError as e:
       raise RuntimeError(f"Invalid JSON in {path}: {e.msg} at line {e.lineno}, col {e.colno}")
   # Langfuse envs
   lf = cfg.get("langfuse") or {}
   if lf.get("enabled"):
       if lf.get("public_key"): os.environ.setdefault("LANGFUSE_PUBLIC_KEY", lf["public_key"])
       if lf.get("secret_key"): os.environ.setdefault("LANGFUSE_SECRET_KEY", lf["secret_key"])
       if lf.get("host"):       os.environ.setdefault("LANGFUSE_HOST", lf["host"])
       os.environ.setdefault("LANGFUSE_PROJECT", lf.get("project", "CustomerSuccessAssistant"))
   # Always export OpenAI key if present (not used by evaluators here, but harmless)
   ok = (cfg.get("openai") or {}).get("api_key")
   if ok: os.environ.setdefault("OPENAI_API_KEY", ok)
   return cfg
def build_llm_from_cfg(cfg):
   from langchain.chat_models import AzureChatOpenAI
   from langchain_openai import ChatOpenAI
   choice = cfg.get("api_choice", "azure").lower()
   if choice.startswith("azure"):
       az = cfg["azure"]
       return AzureChatOpenAI(
           api_key=az["api_key"], azure_endpoint=az["endpoint"], azure_deployment=az["deployment"],
           api_version=az.get("api_version", "2024-02-01"), temperature=0
       )
   else:
       return ChatOpenAI(model=cfg["openai"]["model"], temperature=0)
#def build_langfuse_client():
#   from langfuse import Langfuse
#   return Langfuse()  # reads LANGFUSE_* envs set above
class _NoopObs:
   id = "noop"
class _NoopTrace:
   id = "noop-trace"
   def event(self, **kwargs): pass
   def generation(self, **kwargs): return _NoopObs()
class _NoopLangfuse:
   # new SDK names
   def trace(self, **kwargs): return _NoopTrace()
   def score(self, **kwargs): pass
   def flush(self): pass
   # old SDK names (for our compatibility shims)
   def create_trace(self, **kwargs): return {"id": "noop-trace"}
   def create_event(self, **kwargs): pass
   def create_generation(self, **kwargs): return {"id": "noop"}
   def create_score(self, **kwargs): pass
def build_langfuse_client():
   from langfuse import Langfuse
   # ensure load_api_details() has already set these env vars
   pk = os.getenv("LANGFUSE_PUBLIC_KEY")
   sk = os.getenv("LANGFUSE_SECRET_KEY")
   host = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
   try:
       if pk and sk:
           # ✅ pass credentials explicitly so the client is enabled
           return Langfuse(public_key=pk, secret_key=sk, host=host)
       # No keys? Return a safe no-op client so the app still runs
       return _NoopLangfuse()
   except Exception:
       return _NoopLangfuse()
def get_project_from_cfg(cfg: dict) -> str:
   return (cfg.get("langfuse") or {}).get("project", "CustomerSuccessAssistant")