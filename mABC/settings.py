import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
	"""Minimal .env loader without external deps.

	- Ignores blank lines and comments
	- Does not override existing process environment
	- Supports simple KEY=VALUE lines (optionally quoted)
	"""

	try:
		if not env_path.exists():
			return
		for raw_line in env_path.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key:
				os.environ.setdefault(key, value)
	except Exception:
		# Never break imports because of a local env file.
		return


def _apply_deepseek_compat() -> None:
	"""Allow using DeepSeek OpenAI-compatible API via DEEPSEEK_* vars.

	If user only provides DEEPSEEK_API_KEY in .env, map it to OPENAI_API_KEY and
	set default base_url/model unless user already set OPENAI_* explicitly.
	"""

	deepseek_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
	if deepseek_key and not (os.environ.get("OPENAI_API_KEY") or "").strip():
		os.environ["OPENAI_API_KEY"] = deepseek_key

	deepseek_base = (os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DEEPSEEK_API_BASE_URL") or "").strip()
	if deepseek_base and not (os.environ.get("OPENAI_BASE_URL") or "").strip():
		os.environ["OPENAI_BASE_URL"] = deepseek_base

	deepseek_model = (os.environ.get("DEEPSEEK_MODEL") or "").strip()
	if deepseek_model and not (os.environ.get("OPENAI_MODEL") or "").strip():
		os.environ["OPENAI_MODEL"] = deepseek_model

	# Sensible defaults for DeepSeek OpenAI-compat endpoint.
	if deepseek_key:
		os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
		os.environ.setdefault("OPENAI_MODEL", "deepseek-chat")


def _apply_qwen_compat() -> None:
	"""Allow using Qwen (DashScope) OpenAI-compatible API via QWEN_* vars.

	If the user only provides QWEN_API_KEY/base/model, map them onto OPENAI_*
	so the rest of the code works unchanged. Defaults follow the DashScope
	"compatible mode" endpoint.
	"""

	qwen_key = (os.environ.get("QWEN_API_KEY") or "").strip()
	if qwen_key and not (os.environ.get("OPENAI_API_KEY") or "").strip():
		os.environ["OPENAI_API_KEY"] = qwen_key

	qwen_base = (os.environ.get("QWEN_BASE_URL") or os.environ.get("QWEN_API_BASE_URL") or "").strip()
	if qwen_base and not (os.environ.get("OPENAI_BASE_URL") or "").strip():
		os.environ["OPENAI_BASE_URL"] = qwen_base

	qwen_model = (os.environ.get("QWEN_MODEL") or "").strip()
	if qwen_model and not (os.environ.get("OPENAI_MODEL") or "").strip():
		os.environ["OPENAI_MODEL"] = qwen_model

	if qwen_key:
		os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
		os.environ.setdefault("OPENAI_MODEL", "qwen-plus")


_load_env_file(Path(__file__).resolve().parent / ".env")

# If multiple providers are configured locally, prefer Qwen when present.
# This keeps the workspace default aligned with the "use Qwen" intent, while
# still allowing explicit OPENAI_* env vars to override everything.
_apply_qwen_compat()
_apply_deepseek_compat()


# Prefer environment variables so the repo can run without hardcoding secrets.
# Example:
#   export OPENAI_API_KEY=...
#   export OPENAI_MODEL=gpt-4o-mini
#   export OPENAI_BASE_URL=https://api.openai.com/v1

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "10"))
OPENAI_RETRY_SLEEP = int(os.getenv("OPENAI_RETRY_SLEEP", "30"))
# OPENAI_MODEL = "gpt-3.5-turbo"
# OPENAI_MODEL = "gpt-4"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

# AGENT_STATUS_START = "Start"
# AGENT_STATUS_RE = "Reason"
# AGENT_STATUS_ACT = "Act"
# AGENT_STATUS_FINISH = "Finish"

# STOP_WORDS_REACT = "\nObservation"
# STOP_WORDS_NONE = ""

# ACTION_FAILURE = "action执行失败"
# DEBUG = False


# TOT_CHILDREN_NUM = 1

# TOT_MAX_DEPTH = 15

# # DEFAULT_MODEL = "gpt-3.5-turbo"  # gpt-3.5 -turbo-16k-0613
# # DEFAULT_MODEL = "gpt-4"  # gpt-3.5-turbo-16k-0613
# DEFAULT_MODEL = "gpt-3.5-turbo-0125"

