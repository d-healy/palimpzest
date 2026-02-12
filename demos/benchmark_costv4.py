import os
import time
import math
import random
import functools
import httpx
import threading
import shutil
import tempfile
import litellm
import palimpzest as pz
from palimpzest.policy import MinCost
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ============================================================
# 0) CONFIG FLAGS
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG|INFO|WARN|ERROR
DEBUG_COST_LINES = os.getenv("DEBUG_COST_LINES", "0") == "1"

# Router controls
ROUTER_RETRIES = int(os.getenv("ROUTER_RETRIES", "5"))
ROUTER_TIMEOUT_S = int(os.getenv("ROUTER_TIMEOUT_S", "90"))
INITIAL_BACKOFF_S = float(os.getenv("INITIAL_BACKOFF_S", "0.5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1.8"))
MAX_BACKOFF_S = float(os.getenv("MAX_BACKOFF_S", "12.0"))

# Concurrency
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "4"))  # reduced default
BASELINE_MAX_WORKERS = int(os.getenv("BASELINE_MAX_WORKERS", "6"))

# Workload limiting
FILE_LIMIT = int(os.getenv("FILE_LIMIT", "200"))  # limit to 200 files by default

# ============================================================
# 1) SETUP
# ============================================================
DATASET_PATH = "/home/ubuntu/palimpzest-main/testdata/enron-eval/"
if not os.path.exists(DATASET_PATH):
    print(f"⚠️ Warning: {DATASET_PATH} not found.")

os.environ["LITELLM_LOG"] = "ERROR"
litellm.disable_cost_calculation = True

# Shared HTTP client (self-signed env)
litellm.client_session = httpx.Client(
    verify=False,
    timeout=httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=60.0)
)

# Proxy creds (RealLM - OpenAI-compatible)
os.environ.setdefault("OPENAI_API_KEY", "user-api-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://reallm-proxy.po-cto.dell.com/v1")

# ============================================================
# 2) PRICING MODEL (per 1,000,000 tokens)  — tune as needed
# ============================================================
PRICING = {
    "mistral": {"input": 0.15, "output": 0.15},  # cheap for filter
    "llama":   {"input": 0.12, "output": 0.12},  # RealLM LLaMA-3.1-8B baseline
    "gpt":     {"input": 1.20, "output": 1.20},  # leave GPT slightly < Gemma to prefer GPT
    "gemma":   {"input": 1.70, "output": 1.70},  # a bit > GPT to encourage GPT for mapping
}

# ============================================================
# 2a) BACKEND MODELS (from your /models endpoint)
# ============================================================
BACKEND_MODELS = {
    "MISTRAL_7B":      "openai/mistralai-mistral-7b-instruct-v0-2",
    "LLAMA_8B":        "openai/meta-llama-meta-llama-3-1-8b-instruct",  # RealLM LLaMA for fallback
    "GEMMA_27B":       "openai/google-gemma-3-27b-it",
    "GPT_OSS_120B":    "openai/openai-gpt-oss-120b",
}

# ============================================================
# 2b) ROUTER MAP: Palimpzest canonical ID -> your proxy backend
#  - Use PZ canonical IDs only in Palimpzest configs.
# ============================================================
INTERNAL_MAP = {
    # FILTER canonical -> Mistral first
    "openai/gpt-4.1-nano-2025-04-14": BACKEND_MODELS["MISTRAL_7B"],

    # MAPPING canonicals -> GPT/Gemma only (mapping supports GPT/Gemma)
    "openai/gpt-4o-2024-08-06":       BACKEND_MODELS["GPT_OSS_120B"],   # prefer GPT-style
    "openai/gpt-4o-mini-2024-07-18":  BACKEND_MODELS["GPT_OSS_120B"],   # also GPT-style
}

# ============================================================
# MODEL PRICING MAP (robust -> no substring checks)
# ============================================================
MODEL_PRICING_MAP = {
    BACKEND_MODELS["MISTRAL_7B"]:   "mistral",
    BACKEND_MODELS["LLAMA_8B"]:     "llama",
    BACKEND_MODELS["GEMMA_27B"]:    "gemma",
    BACKEND_MODELS["GPT_OSS_120B"]: "gpt",
}

# ============================================================
# PALIMPZEST MODELS (canonical enum values ONLY)
# ============================================================
PZ_CANONICAL_MODELS = [
    "openai/gpt-4.1-nano-2025-04-14",  # routes to Mistral 7B (filter)
    "openai/gpt-4o-2024-08-06",        # routes to GPT-like (mapping)
    "openai/gpt-4o-mini-2024-07-18",   # routes to GPT-like (mapping)
]

# ============================================================
# Fallback chains per task-type (by canonical)
# ============================================================
FALLBACKS = {
    # FILTER: Mistral -> LLaMA -> Gemma
    "openai/gpt-4.1-nano-2025-04-14": [
        BACKEND_MODELS["MISTRAL_7B"],
        BACKEND_MODELS["LLAMA_8B"],
        BACKEND_MODELS["GEMMA_27B"],
    ],
}

# ============================================================
# Per-backend timeout overrides (seconds)
# ============================================================
BACKEND_TIMEOUTS = {
    BACKEND_MODELS["MISTRAL_7B"]:   int(os.getenv("TIMEOUT_MISTRAL", "120")),
    BACKEND_MODELS["LLAMA_8B"]:     int(os.getenv("TIMEOUT_LLAMA", "90")),
    BACKEND_MODELS["GEMMA_27B"]:    int(os.getenv("TIMEOUT_GEMMA", "90")),
    BACKEND_MODELS["GPT_OSS_120B"]: int(os.getenv("TIMEOUT_GPT", "90")),
}

# ============================================================
# 3) LOGGING
# ============================================================
LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
CUR_LEVEL = LEVELS.get(LOG_LEVEL.upper(), 20)

def log(level: str, msg: str):
    if LEVELS.get(level, 100) >= CUR_LEVEL:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {level:<5} | {msg}")

# ============================================================
# 4) COST TRACKER
# ============================================================
class CostTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.processed = 0
        self.calls_by_model = defaultdict(int)
        self.tokens_by_model = defaultdict(int)
        self.cost_by_model = defaultdict(float)

    def add(self, model_id, usage):
        if not model_id:
            return
        in_tok = 0
        out_tok = 0
        if usage:
            # usage may be a dict or a pydantic-ish object
            try:
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
            except Exception:
                pass
            in_tok = int((usage or {}).get("prompt_tokens", 0) or 0)
            out_tok = int((usage or {}).get("completion_tokens", 0) or 0)

        total_tok = in_tok + out_tok
        with self._lock:
            self.total_tokens += total_tok
            model_key = MODEL_PRICING_MAP.get(model_id, None)
            if not model_key:
                # Default to gemma if unknown (conservative)
                model_key = "gemma"

            if DEBUG_COST_LINES:
                log("DEBUG", f"[COST DEBUG] model={model_id} as={model_key} tok={total_tok} (in={in_tok}, out={out_tok})")

            rates = PRICING[model_key]
            cost = (in_tok / 1_000_000 * rates["input"]) + (out_tok / 1_000_000 * rates["output"])
            self.total_cost += cost
            self.processed += 1

            self.calls_by_model[model_id] += 1
            self.tokens_by_model[model_id] += total_tok
            self.cost_by_model[model_id] += cost

tracker = CostTracker()

# ============================================================
# 5) ROUTER (HARDENED with backoff + fallbacks + prompt rewrite)
#     + AUDIT + PER-BACKEND TIMEOUTS
# ============================================================
_real_completion = litellm.completion

# ---- Router audit structures ----
ROUTER_AUDIT = {
    "attempts_by_canonical": defaultdict(lambda: defaultdict(int)),  # canonical -> backend -> attempts
    "success_by_canonical": defaultdict(lambda: defaultdict(int)),   # canonical -> backend -> successes
    "fail_by_canonical": defaultdict(lambda: defaultdict(int)),      # canonical -> backend -> failures
    "last_error_by_backend": defaultdict(str),                       # backend -> last error
}
CANONICAL_CALLS = defaultdict(int)

def _is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    retryable_terms = [
        "rate limit", "429", "too many requests",
        "timeout", "timed out",
        "connection reset", "connection aborted", "server disconnected",
        "temporarily unavailable", "bad gateway", "502", "503", "504"
    ]
    return any(t in msg for t in retryable_terms)

def _sleep_backoff(attempt: int):
    # Exponential + jitter
    delay = min(MAX_BACKOFF_S, INITIAL_BACKOFF_S * (BACKOFF_FACTOR ** max(0, attempt - 1)))
    delay += random.uniform(0, 0.2 * delay)
    time.sleep(delay)

def _minimal_filter_prompt_rewrite(messages):
    """Inject a tiny system directive to reduce verbosity / enforce concise classification."""
    if not isinstance(messages, list):
        return messages
    sys_hint = {
        "role": "system",
        "content": "You are a concise binary content assessor. Answer minimally and consistently."
    }
    if len(messages) == 0 or messages[0].get("role") != "system":
        return [sys_hint] + messages
    return messages

@functools.wraps(_real_completion)
def router_completion(*args, **kwargs):
    """
    Hardened router:
      - Resolves Palimpzest canonical IDs to backend models.
      - Applies task-aware fallback lists.
      - Retries with exponential backoff on retryable errors.
      - Sanitizes messages to avoid NoneType issues.
      - Adds minimal filter prompt rewrite.
      - Records detailed audit info and per-canonical counts.
      - Applies per-backend timeouts when calling the provider.
    """
    requested_canonical = kwargs.get("model")

    # Normalize/clean messages early
    messages = kwargs.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    cleaned_msgs = []
    for m in messages:
        role = (m or {}).get("role", "user") or "user"
        content = (m or {}).get("content", "")
        if isinstance(content, list):
            content = "".join([x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"])
        if content is None:
            content = ""
        cleaned_msgs.append({"role": role, "content": content})
    kwargs["messages"] = cleaned_msgs

    # Determine fallback chain by canonical (task-aware)
    fallback_chain = FALLBACKS.get(requested_canonical)
    primary_backend = INTERNAL_MAP.get(requested_canonical, requested_canonical)
    if not fallback_chain:
        fallback_chain = [primary_backend]
    else:
        if primary_backend != fallback_chain[0]:
            fallback_chain = [primary_backend] + [m for m in fallback_chain if m != primary_backend]

    # Count one call for this canonical (one logical request)
    CANONICAL_CALLS[requested_canonical] += 1

    # Decide if this is a FILTER call (we only rewrite in that case)
    is_filter = (requested_canonical == "openai/gpt-4.1-nano-2025-04-14")
    if is_filter:
        kwargs["messages"] = _minimal_filter_prompt_rewrite(kwargs["messages"])

    # Stability defaults
    kwargs["ssl_verify"] = False
    # We'll override per-backend below
    kwargs["timeout"] = ROUTER_TIMEOUT_S
    # Let us control retries manually
    kwargs["num_retries"] = 0
    kwargs.pop("reasoning_effort", None)

    last_err = None
    canonical = requested_canonical

    for backend_model in fallback_chain:
        attempt = 0
        while attempt < ROUTER_RETRIES:
            attempt += 1
            # record attempt before trying
            ROUTER_AUDIT["attempts_by_canonical"][canonical][backend_model] += 1
            try:
                use_kwargs = dict(kwargs)
                use_kwargs["model"] = backend_model
                use_kwargs["timeout"] = BACKEND_TIMEOUTS.get(backend_model, ROUTER_TIMEOUT_S)

                log("DEBUG", f"router -> model={backend_model} attempt={attempt}/{ROUTER_RETRIES}, timeout={use_kwargs['timeout']}s")
                resp = _real_completion(*args, **use_kwargs)

                # success path
                usage = getattr(resp, "usage", None)
                try:
                    if hasattr(usage, "model_dump"):
                        usage = usage.model_dump()
                except Exception:
                    pass
                tracker.add(backend_model, usage)

                ROUTER_AUDIT["success_by_canonical"][canonical][backend_model] += 1
                return resp

            except Exception as e:
                last_err = e
                ROUTER_AUDIT["fail_by_canonical"][canonical][backend_model] += 1
                ROUTER_AUDIT["last_error_by_backend"][backend_model] = str(e)

                retryable = _is_retryable_error(e)
                lvl = "WARN" if retryable else "ERROR"
                log(lvl, f"router error: model={backend_model} attempt={attempt} err={e}")
                if not retryable:
                    break  # go to next backend
                if attempt < ROUTER_RETRIES:
                    _sleep_backoff(attempt)
                # else exhausted retries; move to next fallback

        log("WARN", f"backend failed after retries -> {backend_model}, trying next fallback...")

    log("ERROR", f"all backends failed for canonical={requested_canonical}. Raising last error.")
    raise last_err if last_err else RuntimeError("Unknown router failure (no exception captured)")

# Monkey patch
litellm.completion = router_completion

# ============================================================
# 6) HEALTH WARM-UP (optional but helpful)
# ============================================================
def health_ping(model_id: str):
    try:
        litellm.completion(
            model=model_id,
            messages=[{"role":"user","content":"ping"}],
            timeout=min(10, BACKEND_TIMEOUTS.get(model_id, ROUTER_TIMEOUT_S))
        )
        log("INFO", f"Health: {model_id} OK")
    except Exception as e:
        log("WARN", f"Health: {model_id} failed: {e}")

def warm_up_backends():
    seen = set()
    # Cover primary mappings
    for m in INTERNAL_MAP.values():
        if m not in seen:
            seen.add(m); health_ping(m)
    # Cover fallbacks
    for chain in FALLBACKS.values():
        for m in chain:
            if m not in seen:
                seen.add(m); health_ping(m)

# ============================================================
# 7) COST-INSENSITIVE BASELINE (Gemma everywhere)
# ============================================================
def _list_txt_files(path):
    try:
        files = [f for f in os.listdir(path) if f.lower().endswith(".txt")]
        files.sort()
        return files
    except Exception as e:
        log("ERROR", f"reading DATASET_PATH failed: {e}")
        return []

def _build_subset_dir(src_dir, files, limit):
    if limit <= 0 or len(files) <= limit:
        return src_dir, len(files)
    sub = tempfile.mkdtemp(prefix="enron_subset_")
    count = 0
    for f in files[:limit]:
        src = os.path.join(src_dir, f)
        dst = os.path.join(sub, f)
        try:
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
            count += 1
        except Exception as e:
            log("WARN", f"subset copy failed for {f}: {e}")
    return sub, count

def process_one_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        prompt = f"Does this discuss bankruptcy? If yes, extract summary. Email: {content[:1500]}"
        litellm.completion(
            model=BACKEND_MODELS["GEMMA_27B"],
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
    except Exception as e:
        log("WARN", f"baseline file failed: {os.path.basename(path)} err={e}")

def run_baseline_script():
    log("INFO", "🏃 STARTING: Manual Baseline (Gemma Everywhere)...")
    tracker.reset()
    start_time = time.time()

    files = _list_txt_files(DATASET_PATH)
    subset_dir, used = _build_subset_dir(DATASET_PATH, files, FILE_LIMIT)
    log("INFO", f"Baseline using {used} files from: {subset_dir}")

    paths = [os.path.join(subset_dir, f) for f in _list_txt_files(subset_dir)]
    with ThreadPoolExecutor(max_workers=BASELINE_MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_file, p) for p in paths]
        for _ in as_completed(futures):
            pass

    return time.time() - start_time, tracker.total_tokens, tracker.total_cost

# ============================================================
# 8) PALIMPZEST (MinCost policy + canonical IDs + reduced parallel)
# ============================================================
def run_palimpzest():
    log("INFO", "🏃 STARTING: Palimpzest Optimizer (Mistral→LLaMA→Gemma for filter; GPT→Gemma for mapping)...")
    tracker.reset()
    start_time = time.time()

    files = _list_txt_files(DATASET_PATH)
    subset_dir, used = _build_subset_dir(DATASET_PATH, files, FILE_LIMIT)
    log("INFO", f"Palimpzest using {used} files from: {subset_dir}")

    ds = pz.TextFileDataset(id="enron_eval", path=subset_dir)

    # FILTER (binary heuristic)
    ds = ds.sem_filter("The email discusses bankruptcy or liquidation proceedings.")

    # MAPPING (schema)
    ds = ds.sem_map([
        {"name": "summary", "type": str, "desc": "Summary"},
        {"name": "sender",  "type": str, "desc": "Sender"},
    ])

    config = pz.QueryProcessorConfig(
        policy=MinCost(),
        available_models=PZ_CANONICAL_MODELS,
        processing_strategy="no_cache",
        verbose=True,
        max_parallel=MAX_PARALLEL,
    )
    log("INFO", f"Models Palimpzest will consider: {config.available_models}")

    try:
        ds.run(config)
    except Exception as e:
        log("ERROR", f"[PZ ERROR] {e}")

    return time.time() - start_time, tracker.total_tokens, tracker.total_cost

# ============================================================
# 9) MAIN / RESULTS
# ============================================================
if __name__ == "__main__":
    # Routing setup visibility
    log("INFO", "Routing setup:")
    for c, b in INTERNAL_MAP.items():
        log("INFO", f"  canonical {c} -> primary backend {b}")
    for c, chain in FALLBACKS.items():
        log("INFO", f"  canonical {c} fallbacks -> {chain}")
    log("INFO", f"Backend timeouts (s): {BACKEND_TIMEOUTS}")

    # Optional warm-up (helps catch misconfig before workload)
    warm_up_backends()

    base_time, base_tok, base_cost = run_baseline_script()
    pz_time, pz_tok, pz_cost = run_palimpzest()

    print("\n" + "=" * 72)
    print("💰 FINAL COST COMPARISON")
    print("=" * 72)
    print(f"{'METRIC':<20} | {'BASELINE (Gemma)':<22} | {'PALIMPZEST':<22}")
    print("-" * 72)
    print(f"{'Time (s)':<20} | {base_time:<22.2f} | {pz_time:<22.2f}")
    print(f"{'Total Tokens':<20} | {base_tok:<22} | {pz_tok:<22}")
    print("-" * 72)
    print(f"{'ESTIMATED COST':<20} | ${base_cost:<21.4f} | ${pz_cost:<21.4f}")

    print("\nPer-model usage summary:")
    for mid, cnt in tracker.calls_by_model.items():
        print(f"  {mid}: {cnt} calls, {tracker.tokens_by_model[mid]} tokens, ${tracker.cost_by_model[mid]:.4f}")

    print("\nPer-canonical call counts:")
    for k, v in CANONICAL_CALLS.items():
        print(f"  {k}: {v} calls")

    print("\nRouter audit (attempts/success/fail per canonical/backend):")
    for canonical, backends in ROUTER_AUDIT["attempts_by_canonical"].items():
        print(f"  Canonical: {canonical}")
        for backend, attempts in backends.items():
            succ = ROUTER_AUDIT["success_by_canonical"][canonical].get(backend, 0)
            fail = ROUTER_AUDIT["fail_by_canonical"][canonical].get(backend, 0)
            last_err = ROUTER_AUDIT["last_error_by_backend"].get(backend, "")
            summary = f"    - {backend}: attempts={attempts}, success={succ}, fail={fail}"
            if fail and last_err:
                summary += f", last_err='{last_err[:140]}...'"
            print(summary)

    savings = base_cost - pz_cost
    if savings > 0:
        pct = (savings / base_cost * 100) if base_cost > 0 else 0.0
        print(f"\n🏆 WINNER: Palimpzest saved ${savings:.4f} ({pct:.1f}%)")
    else:
        print(f"\n❌ Palimpzest was more expensive by ${abs(savings):.4f}")
    print("=" * 72)
