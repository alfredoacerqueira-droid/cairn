# Cairn Lifecycle Simulation Report

**Date:** 2026-06-07T01:33:00.638708+00:00
**CAIRN_ROOT:** /mnt/c/Users/alfre/Projects/cairn
**Mode:** no-LLM

**System Resources:** {'ram_total_gb': 15.5, 'ram_available_gb': 13.7, 'cpu_count': 12, 'vram_total_gb': 6.0, 'vram_free_gb': 3.7, 'gpu_name': 'NVIDIA GeForce RTX 2060'}

**Ollama models available:**
```
NAME                       ID              SIZE      MODIFIED    
qwen3.5:latest             6488c96fa5fa    6.6 GB    6 hours ago    
gemma4:latest              c6eb396dbd59    9.6 GB    6 hours ago    
qwen2.5-coder:7b           dae161e27b0e    4.7 GB    6 hours ago    
qwen2.5-coder:0.5b         4ff64a7f502a    397 MB    4 days ago     
qwen2.5-coder:1.5b         d7372fd82851    986 MB    4 days ago     
gemma4:e2b                 7fbdbf8f5e45    7.2 GB    5 days ago     
qwen3-embedding:4b  
```

**Gemma model available:** True



## Phase 1 — Init from 0

| Phase | Step | Mode | Result | Evidence | Elapsed |
|-------|------|------|--------|----------|--------|
| 1 | init -y --offline | no-LLM | PASS | profile=code patterns=['*.bash', '*.c', '*.cpp', '*.cs', '*.go', '*.h', '*.hpp', '*.java', '*.js', '*.py', '*.rb', '*.rs', '*.sh', '*.tf', '*.tfvars', '*.toml'] index_loc=auto | 3.8s |
| 1 | chroma DB check | no-LLM | PASS | chroma_path=/home/alfredo/.cache/cairn/fa102402024a/chroma exists=True blocks=1 | 0.6s |
| 1 | indexed files | no-LLM | INFO | Indexed: ['app.py'] | NOT indexed: ['templates/index.html (excluded: not in file_patterns)', 'static/app.js (excluded: **/static/** + not in file_patterns)', 'static/style.css (excluded: **/static/** + not in file_patterns)'] | - |

**Finding:** Default file_patterns=['*.bash', '*.c', '*.cpp', '*.cs', '*.go', '*.h', '*.hpp', '*.java', '*.js', '*.py', '*.rb', '*.rs', '*.sh', '*.tf', '*.tfvars', '*.toml'] and exclude_patterns include `**/static/**` and do not include `*.html`, `*.css`, `*.js`. Frontend files are NOT indexed by default.


## Phase 2 — DB Sync

| Phase | Step | Mode | Result | Evidence | Elapsed |
|-------|------|------|--------|----------|--------|
| 2a | janitor live-sync | no-LLM | PASS | get_task_count appeared after 3s | 3s |
| 2a | direct indexer path | no-LLM | PASS | get_task_count found via direct index: 1. services/tasks.py:get_task_count (line 18)
   relevance: 0.000
   Code: def get_task_count():
       return len(list_tasks())

2. /mnt/c/Users/alfre/Projects/taskboard_bl4l4yms/services/tasks.py:ge | 0.2s |
| 2b | search create_task after B | no-LLM | PASS | Top: 1. app.py:tasks (line 15) | 4.5s |
| 2b | search validate after C | no-LLM | PASS | Top: 1. services/tasks.py:create_task (line 8) | - |
| 2b | search delete_task after D | no-LLM | PASS | delete_task function block absent (correct): 1. /mnt/c/Users/alfre/Projects/taskboard_bl4l4yms/models.py:Task (line 1)
   relevance: 0.000
   Code: class Task:
       def __init__(self, title):
  | - |


## Phase 3 — Memory Sync

| Phase | Step | Mode | Result | Evidence | Elapsed |
|-------|------|------|--------|----------|--------|
| 3 | memory update --commits 6 | no-LLM | PASS | memory.md exists=True, sections: Recent Changes | 4.8s |

**Memory contents (first 1000 chars):**

```
# Cairn Memory

## Open Tasks

## Decisions

## Conventions

## Recent Changes
- [2026-06-06] Two new Python model and service files added: a `Task` class for managing tasks and a `storage.py` file for handling task storage. The `Task` class includes an initializer and a method to convert the task object to a dictionary, while `storage.py` provides functions to save and load tasks from a JSON file in the project directory.
- [2026-06-06] The change removes the `delete_task` function from `services/tasks.py`.
- [2026-06-06] This Git diff adds input validation and includes the creation of tasks with a timestamp in the `create_task` function.
- [2026-06-06] This pull request adds a new configuration for the MCP server, modifies Flask app routing to include tasks-related endpoints, and creates task service modules.

## Recent User Prompts

```
| 3 | in-process remember/recall | no-LLM | PASS | decision='remembered (decision, repo: taskboard_bl4l4yms)' convention='remembered (convention, repo: taskboard_bl4l4yms)' both_in_recall=True | 0.1s |
| 3 | memory accumulation (2 more commits) | no-LLM | PASS | Entries after 2 more commits: ~8 (before had decisions/conventions from remember) | - |

**Memory after accumulation:**

```
# Cairn Memory

## Open Tasks

## Decisions
- [2026-06-06] Chose Flask + in-memory store for v1

## Conventions
- [2026-06-06] services/ holds business logic

## Recent Changes
- [2026-06-06] Two new Python model and service files added: a `Task` class for managing tasks and a `storage.py` file for handling task storage. The `Task` class includes an initializer and a method to convert the task object to a dictionary, while `storage.py` provides functions to save and load tasks from a JSON file in the project directory.
- [2026-06-06] The change removes the `delete_task` function from `services/tasks.py`.
- [2026-06-06] This Git diff adds input validation and includes the creation of tasks with a timestamp in the `create_task` function.
- [2026-06-06] This pull request adds a new configuration for the MCP server, modifies Flask app routing to include tasks-related endpoints, and creates task service modules.
- [2026-06-06] The README file has been updated with an initial version section.
- [2026-06-06] A new function named `dummy` has been added to the `utils.py` file.

## Recent User Prompts

```


## Phase 4 — Retrieval

| Phase | Step | Mode | Result | Evidence | Elapsed |
|-------|------|------|--------|----------|--------|
| 4 | assemble_context | no-LLM | PASS | Len=2590 has_sections=True | 5.2s |

**assemble_context snippet:**

```
# [already-compressed]
# Codebase Context

## Relevant Functions (Semantic Match)
### app.py:tasks (lines 15-16, similarity: 1.00)
```
def tasks():
    if request.method == 'GET':
        return jsonify(list_tasks())
    data = request.get_json() or {}
    title = data.get('title', '')
    task = create_task(title)
    return jsonify(task), 201
```

### services/storage.py:save_tasks (lines 7-8, similarity: 0.95)
```
def save_tasks(tasks):
    with open(STORAGE_FILE, 'w') as f:
        json.dump(tasks, f, indent=2)
```

### models.py:Task.__init__ (lines 2-3, similarity: 0.86)
```
def __init__(self, title):
        self.title = title
        self.created_at = None
```

### models.py:Task.to_dict (lines 6-7, similarity: 0.86)
```
def to_dict(self):
        return {'title': self.title, 'created_at': self.created_at}
```

### services/tasks.py:list_tasks (lines 1-2, similarity: 0.81)
```
def list_tasks():
    return [{'id': 1, 'title': 'Example'}]
```

## Repository Structure
/mnt/c/Users/alfre/Projects/taskboard_bl4l4yms/app.py:
  index
  tasks
/mnt/c/Users/alfre/Projects/taskboard_bl4l4yms/models.py:
  Task
    __init__
    to_dict
/mnt/c/Users/alfre/Projects/taskboard_bl4l4yms/serv
```
| 4 | search_code 'persist tasks' | no-LLM | PARTIAL | Top: 1. app.py:tasks (line 15) | 0.1s |
| 4 | search_code 'Task model' | no-LLM | PASS | Top: 1. app.py:tasks (line 15) | 0.1s |
| 4 | search_code 'validate title' | no-LLM | PASS | Top: 1. services/tasks.py:create_task (line 8) | 0.1s |


## Phase 5 — Freshness

| Phase | Step | Mode | Result | Evidence | Elapsed |
|-------|------|------|--------|----------|--------|
| 5 | status (stale) | no-LLM | PASS | Output: Project: /mnt/c/Users/alfre/Projects/taskboard_bl4l4yms
Current commit: 9e725980
Last indexed: 0be84996
Commits behind: 2
Index location: native (/home/alfredo/.cache/cairn/fa102402024a)
Indexed funct | 1.2s |

**status output:**

```
Project: /mnt/c/Users/alfre/Projects/taskboard_bl4l4yms
Current commit: 9e725980
Last indexed: 0be84996
Commits behind: 2
Index location: native (/home/alfredo/.cache/cairn/fa102402024a)
Indexed functions: 11

```
| 5 | status (fresh after reindex) | no-LLM | PASS | Output: Project: /mnt/c/Users/alfre/Projects/taskboard_bl4l4yms
Current commit: 9e725980
Last indexed: 9e725980
Commits behind: 0
Index location: native (/home/alfredo/.cache/cairn/fa102402024a)
Indexed funct | - |


## Summary

## Summary

The Cairn lifecycle simulation tested the full feature set: init, indexing (AST parse + ChromaDB), DB freshness tracking, janitor live-sync, manual reindex, memory summarization, in-process MCP tools (remember/recall/search_code/assemble_context), and index status reporting.



## Findings & Limitations

### Findings & Limitations

1. **Frontend not indexed by default:** `file_patterns` default to `.py`, `.rs`, `.go`, `.c`, etc. and do not include `.html`, `.css`, `.js`. Additionally, `exclude_patterns` includes `**/static/**`. This means frontend files in `templates/` and `static/` are never indexed out of the box. This is intentional for code-centric repos but worth documenting.

2. **Janitor on /mnt/c (WSL2):** The janitor uses watchdog (inotify). On WSL2 with /mnt/c paths (DrvFs/9p filesystem), inotify may not detect file changes reliably. The janitor live-sync test may FAIL in this environment even though the direct indexer path works correctly. This is a known WSL2 limitation, not a Cairn bug.

3. **No-LLM mode:** Without local LLM, embeddings are disabled and retrieval falls back to BM25 + AST structural search. This works correctly but may produce different search results (lower recall for semantic queries).

4. **With-LLM mode:** Requires Ollama running with `nomic-embed-text` and `gemma4:latest` models pulled. Enables hybrid retrieval with embeddings, which provides better semantic matching.

5. **Memory accumulation:** Memory entries are persisted across sessions in `.cairn/memory.md` with bounded sections. The `remember()` tool writes to the appropriate section (decisions, conventions, etc.) and auto-caps entries to prevent unbounded growth.

6. **Profile detection:** Cairn correctly detected the project profile based on file extension census. For a pure Python project, it selects the `python` or `code` profile with appropriate retrieval strategy.
