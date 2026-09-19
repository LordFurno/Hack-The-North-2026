from core import Detection, Entity
from pathlib import Path
import json, os, re, sys, threading

try: #The key lives in .env beside the yibu examples, which is where the team keeps it.
    from dotenv import load_dotenv #Optional: without it, export YIBU_API_KEY yourself
    load_dotenv()
except ImportError:
    pass

#Omni's first job: name a thing, once, at birth. It never detects and it is never in the
#settle path -- a 3-4 s call per settle would queue settles behind each other on stale
#frames. The entity is minted, tracked and drawn immediately as "unknown" and the label
#patches in when the call returns, so tracking never waits on the network.
#
#Every call goes through yibu_http.chat_completion, never a hand-rolled request, because
#that is what appends the token record to artifacts/yibu_api_calls.jsonl. A call this
#file makes by itself is a call nobody can account for afterwards.

EXAMPLES_DIR = Path(__file__).resolve().parent / "yibuapi_examples_20260918_v01"

BASE_URL = os.environ.get("YIBU_BASE_URL", "https://yibuapi.com/v1")
MODEL = os.environ.get("YIBU_MODEL", "qwen3.5-omni-flash")
API_KEY_ENV = "YIBU_API_KEY"
PURPOSE = "spatial_memory_label" #How this project's spend shows up in the ledger
MAX_TOKENS = 64

PROMPT = ("This is a crop of one object on a desk, seen from directly overhead. "
          "Reply with JSON only, no prose and no code fence: "
          '{"label": "<one or two common words>", "isContainer": <true|false>}. '
          "isContainer is true only if things can be put INSIDE it, like a mug, a tin or "
          "a box, and false for a solid object something can merely sit on top of.")

_lock = threading.Lock()
_labelled: set[str] = set() #Entity ids already asked about. Cached forever, never re-called


def apiKey() -> str | None:
    return os.environ.get(API_KEY_ENV) or None


def auditLog() -> Path:
    #One ledger, wherever the process was launched from. YIBU_AUDIT_LOG is a RELATIVE
    #path in .env, so it resolves against the working directory: live.py run from the
    #repo root would write a second ledger that summarize_usage.py never reads, and the
    #spend would look like it never happened. An absolute override is still honoured.
    env = os.environ.get("YIBU_AUDIT_LOG")
    if env and Path(env).is_absolute():
        return Path(env)
    return EXAMPLES_DIR / "artifacts" / "yibu_api_calls.jsonl"


def yibuHttp():
    #Imported late and by path: the examples ship as scripts beside the repo, not as an
    #installed package, and importing them at module load would make every test that
    #touches omni depend on httpx being present.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    import yibu_http
    return yibu_http


def ask(cropPath: str, key: str, model: str = MODEL, baseUrl: str = BASE_URL) -> str:
    #Returns the reply text. The token record lands in the ledger as a side effect of
    #going through their client, which is the whole reason to go through it.
    http = yibuHttp()
    text, _response, _record = http.chat_completion(
        api_key=key, model=model,
        messages=http.build_omni_messages(PROMPT, image=Path(cropPath)),
        purpose=PURPOSE, base_url=baseUrl, max_tokens=MAX_TOKENS, temperature=0.0,
        audit_log=auditLog())
    return text


def parse(text: str) -> tuple[str, bool] | None:
    #Models fence their JSON however they like, so take the first object in the reply.
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    label = str(data.get("label", "")).strip().lower()
    return (label, bool(data.get("isContainer"))) if label else None


def label(e: Entity, det: Detection, worldLock=None, transport=ask):
    #Runs on its own thread. Anything that goes wrong leaves the entity as "unknown",
    #which is a worse answer and not a wrong one: the world model never needed the name.
    key = apiKey()
    if key is None or not det.crop_path or not os.path.exists(det.crop_path):
        return
    try:
        got = parse(transport(det.crop_path, key))
    except Exception as err: #Their client re-raises whatever httpx raised, after auditing
        print(f"[omni] {e.id} stays unknown: {type(err).__name__}: {err}")
        return
    if got is None:
        return

    with (worldLock or _lock): #The settle thread is reading these while we write them
        e.label, e.isContainer = got
    print(f"[omni] {e.id} is a {e.label}" + (" (container)" if e.isContainer else ""))


def labelAsync(e: Entity, det: Detection, worldLock=None, transport=ask):
    #The hook World.onMint fires. One call per new entity, ever.
    if apiKey() is None:
        return
    with _lock:
        if e.id in _labelled:
            return
        _labelled.add(e.id)
    threading.Thread(target=label, args=(e, det, worldLock, transport), daemon=True).start()


def attach(world, worldLock=None):
    #Wire labelling into a world. Without this call nothing here ever runs.
    world.onMint = lambda e, det: labelAsync(e, det, worldLock)
