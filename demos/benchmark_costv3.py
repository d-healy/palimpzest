import os
import time
import functools
import httpx
import threading
import litellm
import palimpzest as pz
from palimpzest.policy import MinCost
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ---------------------------
# 0. CONFIG FLAGS
# ---------------------------
DEBUG_COST_LINES = True
ROUTER_RETRIES = 5
ROUTER_TIMEOUT_S = 90
MAX_PARALLEL = 6

# ---------------------------
# 1. SETUP
# ---------------------------
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

# Proxy creds
os.environ["OPENAI_API_KEY"] = "user-api-key"
os.environ["OPENAI_BASE_URL"] = "https://reallm-proxy.po-cto.dell.com/v1"

# ---------------------------
# 2. PRICING MODEL (per 1,000,000 tokens)
# ---------------------------
PRICING = {
    "mistral": {"input": 0.15, "output": 0.15},  # cheap for filter
    "gpt":     {"input": 1.20, "output": 1.20},  # set slightly < gemma
    "gemma":   {"input": 1.70, "output": 1.70},  # a bit > gpt to encourage GPT for map
}

# ---------------------------
# 2a. BACKEND MODELS (from your /models endpoint)
# ---------------------------
BACKEND_MODELS = {
    "MISTRAL_7B":      "openai/mistralai-mistral-7b-instruct-v0-2",
    "GEMMA_27B":       "openai/google-gemma-3-27b-it",
    "GPT_OSS_120B":    "openai/openai-gpt-oss-120b",
}

# ---------------------------
# 2a. BACKEND MODELS (from your /models endpoint)
# ---------------------------
BACKEND_MODELS = {
    "MISTRAL_7B":      "openai/mistralai-mistral-7b-instruct-v0-2",
    "GEMMA_27B":       "openai/google-gemma-3-27b-it",
    "GPT_OSS_120B":    "openai/openai-gpt-oss-120b",
}

# ---------------------------
# 2b. ROUTER MAP: Palimpzest canonical ID -> your proxy backend
# ---------------------------
INTERNAL_MAP = {
    # Palimpzest canonical IDs (must match the enum you saw in the error)
    "openai/gpt-4.1-nano-2025-04-14": BACKEND_MODELS["MISTRAL_7B"],   # filter
    "openai/gpt-4o-2024-08-06":       BACKEND_MODELS["GEMMA_27B"],    # map baseline
    "openai/gpt-4o-mini-2024-07-18":  BACKEND_MODELS["GPT_OSS_120B"], # your GPT target
}

# ---------------------------
# MODEL PRICING MAP (robust -> no substring checks)
# ---------------------------
MODEL_PRICING_MAP = {
    BACKEND_MODELS["MISTRAL_7B"]:   "mistral",
    BACKEND_MODELS["GEMMA_27B"]:    "gemma",
    BACKEND_MODELS["GPT_OSS_120B"]: "gpt",
}

# ---------------------------
# PALIMPZEST MODELS (use canonical enum values ONLY)
# ---------------------------
PZ_CANONICAL_MODELS = [
    "openai/gpt-4.1-nano-2025-04-14",  # routes to Mistral 7B
    "openai/gpt-4o-2024-08-06",        # routes to Gemma 27B IT
    "openai/gpt-4o-mini-2024-07-18",   # routes to GPT-OSS 120B
]

# ---------------------------
# 3. COST TRACKER
# ---------------------------
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
        if not usage:
            return
        with self._lock:
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            total_tok = in_tok + out_tok
            self.total_tokens += total_tok

            model_key = MODEL_PRICING_MAP.get(model_id, "gemma")

            if DEBUG_COST_LINES:
                print(f"[COST DEBUG] Model used: {model_id}, Classified as: {model_key}, Tokens: {total_tok}")

            rates = PRICING[model_key]
            cost = (in_tok / 1_000_000 * rates["input"]) + (out_tok / 1_000_000 * rates["output"])
            self.total_cost += cost
            self.processed += 1

            self.calls_by_model[model_id] += 1
            self.tokens_by_model[model_id] += total_tok
            self.cost_by_model[model_id] += cost

tracker = CostTracker()

# ---------------------------
# 4. ROUTER
# ---------------------------
_real_completion = litellm.completion

@functools.wraps(_real_completion)
def router_completion(*args, **kwargs):
    requested = kwargs.get("model")

    # Map Palimpzest canonical -> your proxy backend
    if requested in INTERNAL_MAP:
        kwargs["model"] = INTERNAL_MAP[requested]

    # Stability defaults
    kwargs["ssl_verify"] = False
    kwargs.setdefault("num_retries", ROUTER_RETRIES)
    kwargs.setdefault("timeout", ROUTER_TIMEOUT_S)
    kwargs.pop("reasoning_effort", None)

    # Normalize messages: flatten text parts
    if "messages" in kwargs:
        for m in kwargs["messages"]:
            if isinstance(m.get("content"), list):
                m["content"] = "".join(
                    [x.get("text", "") for x in m["content"] if x.get("type") == "text"]
                )

    try:
        resp = _real_completion(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        tracker.add(kwargs["model"], usage)
        return resp

    except Exception as e:
        print(f"[ROUTER ERROR] Model {kwargs.get('model')} failed with: {e}")
        raise   # <- re-raise, don't return None

litellm.completion = router_completion

# ---------------------------
# 5. BASELINE (Gemma everywhere)
# ---------------------------
def process_one_file(f_name):
    try:
        path = os.path.join(DATASET_PATH, f_name)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        prompt = f"Does this discuss bankruptcy? If yes, extract summary. Email: {content[:1500]}"

        litellm.completion(
            model=BACKEND_MODELS["GEMMA_27B"],
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
    except:
        pass

def run_baseline_script():
    print("\n🏃 STARTING: Manual Baseline (Gemma Everywhere)...")
    tracker.reset()
    start_time = time.time()

    files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".txt")]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_one_file, f) for f in files]
        for _ in as_completed(futures):
            pass

    return time.time() - start_time, tracker.total_tokens, tracker.total_cost

# ---------------------------
# 6. PALIMPZEST
# ---------------------------
def run_palimpzest():
    print("\n\n🏃 STARTING: Palimpzest Optimizer (3 Models)...")
    tracker.reset()
    start_time = time.time()

    ds = pz.TextFileDataset(id="enron_eval", path=DATASET_PATH)
    ds = ds.sem_filter("The email discusses bankruptcy or liquidation proceedings.")
    ds = ds.sem_map([
        {"name": "summary", "type": str, "desc": "Summary"},
        {"name": "sender", "type": str, "desc": "Sender"},
    ])

    # ❗ Use canonical Palimpzest IDs here (from the enum in your error)
    config = pz.QueryProcessorConfig(
        policy=MinCost(),
        available_models=PZ_CANONICAL_MODELS,
        processing_strategy="no_cache",
        verbose=True,
        max_parallel=MAX_PARALLEL,
    )
    print("Models Palimpzest will consider:", config.available_models)

    try:
        ds.run(config)
    except Exception as e:
        print(f"[PZ ERROR] {e}")

    return time.time() - start_time, tracker.total_tokens, tracker.total_cost

# ---------------------------
# 7. RESULTS
# ---------------------------
if __name__ == "__main__":
    base_time, base_tok, base_cost = run_baseline_script()
    pz_time, pz_tok, pz_cost = run_palimpzest()

    print("\n" + "=" * 65)
    print("💰 FINAL COST COMPARISON (1000 Files)")
    print("=" * 65)
    print(f"{'METRIC':<20} | {'BASELINE (Gemma)':<20} | {'PALIMPZEST':<20}")
    print("-" * 65)
    print(f"{'Time (s)':<20} | {base_time:<20.2f} | {pz_time:<20.2f}")
    print(f"{'Total Tokens':<20} | {base_tok:<20} | {pz_tok:<20}")
    print("-" * 65)
    print(f"{'ESTIMATED COST':<20} | ${base_cost:<19.4f} | ${pz_cost:<19.4f}")

    print("\nPer-model usage summary:")
    for mid, cnt in tracker.calls_by_model.items():
        print(f"  {mid}: {cnt} calls, {tracker.tokens_by_model[mid]} tokens, ${tracker.cost_by_model[mid]:.4f}")

    savings = base_cost - pz_cost
    if savings > 0:
        pct = (savings / base_cost * 100) if base_cost > 0 else 0.0
        print(f"\n🏆 WINNER: Palimpzest saved ${savings:.4f} ({pct:.1f}%)")
    else:
        print(f"\n❌ Palimpzest was more expensive by ${abs(savings):.4f}")
    print("=" * 65)
