"""Final verification: fastembed-default (NO Ollama) on django — relevance + latency.

Proves Workstream A+B: semantic retrieval works offline via fastembed and latency is
down from the ~19s baseline. Cold-indexes django with fastembed (bge-small, 384d),
then runs representative dev tasks measuring gt-area relevance + warm assemble latency.
"""
import sys, time, logging, shutil
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/alfre/Projects/cairn")
logging.disable(logging.WARNING)

from core.config import Config, save_config
from core.repo import RepoManager, collect_source_files, detect_source_layout
from pipeline.ast_parser import ASTParser
from server.context_assembler import ContextAssembler

DJ = Path("/mnt/c/Users/alfre/Projects/cairn-hardtest/corpus/django")
TASKS = [
    ("how does QuerySet.filter build the query", "filter"),
    ("where is the CSRF token validated", "csrf"),
    ("how does Paginator compute pages", "paginator"),
    ("how are URL patterns resolved to a view", "resolv"),
    ("how does the model save persist a row", "save"),
    ("how is a QuerySet turned into SQL", "compiler"),
    ("how does the template engine render", "template"),
    ("how are forms validated", "forms"),
    ("how does the cache framework get and set", "cache"),
    ("how does signing protect a value", "sign"),
]

# 1. Configure django for fastembed (offline, no Ollama)
roots, patterns = detect_source_layout(DJ)
cairn_dir = DJ / ".cairn"
if cairn_dir.exists():
    shutil.rmtree(cairn_dir)
cfg = Config()
cfg.indexing.source_roots = roots
cfg.indexing.file_patterns = patterns
cfg.profile = "python"
cfg.embeddings_enabled = True
cfg.local_llm.enabled = False
cfg.local_llm.embedder = "fastembed"
save_config(cfg, DJ)

# 2. Cold index with fastembed via the real assembler/indexer path (no Ollama)
print("Indexing django with fastembed (bge-small, no Ollama)...", flush=True)
t0 = time.perf_counter()
asm = ContextAssembler(project_path=DJ)
files = collect_source_files(DJ, cfg.indexing.file_patterns, cfg.indexing.exclude_patterns, roots)
parser = ASTParser()
n = 0
for f in files:
    try:
        asm.vector_indexer.index_ast(parser.parse_file(f))
        n += 1
    except Exception:
        pass
idx_s = time.perf_counter() - t0
print(f"Indexed {n} files in {idx_s:.0f}s", flush=True)

# fresh assembler so it reads the built index
asm = ContextAssembler(project_path=DJ)

# 3. Measure relevance (gt-area in top-5) + warm assemble latency
print("\n=== fastembed-default (offline) on django ===", flush=True)
hits = 0
lat = []
for q, area in TASKS:
    asm.cache = None
    t = time.perf_counter()
    fns = asm.semantic_search(q, top_k=5)
    lat.append((time.perf_counter() - t) * 1000)
    blob = " ".join((f.get("filepath", "") + f.get("function", "")).lower() for f in fns)
    ok = area in blob
    hits += ok
    top = fns[0] if fns else {}
    print(f"  [{'Y' if ok else 'n'}] '{q[:40]}' -> {Path(top.get('filepath','')).name}:{top.get('function','')}", flush=True)

lat.sort()
print(f"\nRelevance (gt-area in top-5): {hits}/{len(TASKS)}", flush=True)
print(f"Warm assemble latency: median={lat[len(lat)//2]:.0f}ms  min={lat[0]:.0f}ms  max={lat[-1]:.0f}ms", flush=True)
print(f"Index time (fastembed, {n} files): {idx_s:.0f}s  | NO Ollama used", flush=True)
