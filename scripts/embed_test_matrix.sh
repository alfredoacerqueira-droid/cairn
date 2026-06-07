#!/usr/bin/env bash
set -uo pipefail

# --- Configurable env vars ---
REPOS="${REPOS:-/mnt/c/Users/alfre/Projects/cairn /mnt/c/Users/alfre/Projects/django /mnt/c/Users/alfre/Projects/testAPI /mnt/c/Users/alfre/Projects/tf-eks /mnt/c/Users/alfre/Projects/rtk /mnt/c/Users/alfre/Projects/csharp-mediatr /mnt/c/Users/alfre/Projects/istioambient}"
MODES="${MODES:-ollama fastembed}"
WIPE_ALL="${WIPE_ALL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAIRN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_FILE="$SCRIPT_DIR/embed_matrix_results.md"
LOG_FILE="$SCRIPT_DIR/embed_matrix_run.log"

# Ensure cairn root is in PYTHONPATH so imports work from any directory
export PYTHONPATH="$CAIRN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG_FILE"
}

# --- Header for results file (create if missing/empty) ---
if [[ ! -f "$RESULTS_FILE" ]] || [[ ! -s "$RESULTS_FILE" ]]; then
    {
        echo "| repo | mode | embedder | dim | files | functions | wall_s | status | note |"
        echo "|------|------|----------|-----|-------|-----------|--------|--------|------|"
    } >"$RESULTS_FILE"
fi

# --- WIPE_ALL: clear entire cairn cache ONCE at start ---
if [[ "$WIPE_ALL" == "1" ]]; then
    echo "WIPE_ALL=1: removing ~/.cache/cairn"
    log "WIPE_ALL=1: rm -rf ~/.cache/cairn"
    rm -rf ~/.cache/cairn
fi

# --- Main loop: each repo × each mode ---
for repo in $REPOS; do
    if [[ ! -d "$repo" ]]; then
        echo "SKIP: directory not found: $repo"
        log "SKIP dir_not_found: $repo"
        continue
    fi

    repo_name="$(basename "$repo")"

    # Compute project-id using cairn's own function
    proj_id=""
    proj_id=$(python3 -c "from core.repo import project_id; print(project_id('$repo'))" 2>>"$LOG_FILE")
    if [[ -z "$proj_id" ]]; then
        # Fallback: replicate the algorithm exactly
        proj_id=$(python3 -c "import hashlib,pathlib; print(hashlib.sha1(str(pathlib.Path('$repo').resolve()).encode()).hexdigest()[:12])" 2>/dev/null)
        if [[ -z "$proj_id" ]]; then
            echo "FAIL: cannot compute project-id for $repo"
            log "FAIL project_id: $repo"
            echo "| $repo_name | - | - | - | - | - | - | FAIL | project-id computation failed |" >>"$RESULTS_FILE"
            continue
        fi
        log "WARN: used fallback project_id for $repo -> $proj_id"
    fi

    for mode in $MODES; do
        echo ""
        echo "============================================================"
        echo "==> REPO: $repo_name  MODE: $mode  PROJECT-ID: $proj_id"
        echo "============================================================"
        log "START repo=$repo_name mode=$mode proj_id=$proj_id"

        # --- Clean this repo's DB ---
        rm -rf ~/.cache/cairn/"$proj_id"
        log "cleaned ~/.cache/cairn/$proj_id"

        # --- Write config.yaml ---
        mkdir -p "$repo/.cairn"
        if [[ "$mode" == "fastembed" ]]; then
            cat >"$repo/.cairn/config.yaml" <<'CONFEOF'
profile: code
embeddings_enabled: true
local_llm:
  enabled: false
  embedder: fastembed
  fastembed_model: BAAI/bge-small-en-v1.5
  embed_device: auto
  embed_threads: 0
indexing:
  store_backend: chroma
  index_location: auto
CONFEOF
        else
            cat >"$repo/.cairn/config.yaml" <<'CONFEOF'
profile: code
embeddings_enabled: true
local_llm:
  enabled: true
  backend: ollama
  embedder: ollama
  embed_model: nomic-embed-text
indexing:
  store_backend: chroma
  index_location: auto
CONFEOF
        fi
        log "wrote $repo/.cairn/config.yaml for mode=$mode"

        # --- Prove active embedder ---
        # Run python imports from CAIRN_ROOT so 'import core' resolves to cairn,
        # not the target repo (which may have its own core/ package shadowing us).
        clear_cache=$(python3 -c "
from core.config import clear_config_cache
from pathlib import Path
clear_config_cache(Path('$repo'))
" 2>>"$LOG_FILE")

        embed_proof_raw=$(python3 -c "
from core.config import load_config
from pipeline.store.embedders import make_embedder
from pathlib import Path
e = make_embedder(load_config(Path('$repo')))
print('EMBPROOF', e.name, e.dim)
" 2>>"$LOG_FILE")
        proof_exit=$?

        if [[ "$proof_exit" -ne 0 ]]; then
            log "EMBPROOF crash exit=$proof_exit"
            # Read last few lines of log for stderr detail
            err_tail=$(tail -n 25 "$LOG_FILE" 2>/dev/null | sed 's/|/\\|/g' | tr '\n' ' ')
            echo "| $repo_name | $mode | - | - | - | - | - | FAIL | embedder proof crashed (exit $proof_exit) |" >>"$RESULTS_FILE"
            continue
        fi

        embedder_name=$(echo "$embed_proof_raw" | grep -oP 'EMBPROOF \K\S+')
        embedder_dim=$(echo "$embed_proof_raw" | grep -oP 'EMBPROOF \S+ \K\d+')

        if [[ -z "$embedder_name" ]] || [[ -z "$embedder_dim" ]]; then
            log "EMBPROOF parse fail raw=$embed_proof_raw"
            echo "| $repo_name | $mode | unknown | ? | - | - | - | FAIL | embedder proof parse failed |" >>"$RESULTS_FILE"
            continue
        fi

        if [[ "$embedder_name" == "placeholder" ]]; then
            echo "  EMBPROOF: placeholder (embeddings OFF) — marking FAIL"
            log "EMBPROOF placeholder for $repo_name $mode"
            echo "| $repo_name | $mode | $embedder_name | $embedder_dim | 0 | 0 | 0 | FAIL | placeholder — embeddings off |" >>"$RESULTS_FILE"
            echo "  -> FAIL: placeholder — embeddings off"
            continue
        fi

        echo "  EMBPROOF: $embedder_name dim=$embedder_dim"
        log "EMBPROOF ok: $embedder_name dim=$embedder_dim"

        # --- Time the reindex ---
        cd "$repo" || {
            log "FAIL cd to $repo for reindex"
            echo "| $repo_name | $mode | $embedder_name | $embedder_dim | - | - | - | FAIL | cd failed for reindex |" >>"$RESULTS_FILE"
            cd "$CAIRN_ROOT"
            continue
        }

        reindex_start=$SECONDS
        reindex_stdout=""
        reindex_stderr=""
        reindex_exit=0

        # Capture stdout+stderr separately
        reindex_stdout=$(cairn reindex --mode full 2>"$repo/.cairn/reindex_stderr.tmp")
        reindex_exit=$?
        wall_s=$((SECONDS - reindex_start))
        reindex_stderr=$(cat "$repo/.cairn/reindex_stderr.tmp" 2>/dev/null || true)
        rm -f "$repo/.cairn/reindex_stderr.tmp"

        cd "$CAIRN_ROOT"

        # Log stderr if non-empty
        if [[ -n "$reindex_stderr" ]]; then
            log "reindex stderr: $reindex_stderr"
        fi
        log "reindex exit=$reindex_exit wall_s=$wall_s"

        # Parse count line: "Indexed N functions from M files"
        func_count=""
        file_count=""
        if echo "$reindex_stdout" | grep -qP 'Indexed \d+ functions from \d+ files'; then
            func_count=$(echo "$reindex_stdout" | grep -oP 'Indexed \K\d+' | head -1)
            file_count=$(echo "$reindex_stdout" | grep -oP 'from \K\d+')
        fi

        echo "  reindex: $func_count functions from $file_count files in ${wall_s}s (exit=$reindex_exit)"

        # --- Determine status and write row ---
        if [[ "$reindex_exit" -eq 0 ]] && [[ -n "$func_count" ]] && [[ -n "$file_count" ]]; then
            status="OK"
            note=""
        else
            status="FAIL"
            note=""
            if [[ "$reindex_exit" -ne 0 ]]; then
                note="reindex exit=$reindex_exit; "
            fi
            if [[ -z "$func_count" ]] || [[ -z "$file_count" ]]; then
                note="${note}no count line parsed; "
            fi
            # Snip last ~25 lines of stderr for the note
            if [[ -n "$reindex_stderr" ]]; then
                err_snip=$(echo "$reindex_stderr" | tail -n 25 | sed 's/|/\\|/g' | tr '\n' '; ')
                note="${note}stderr: ${err_snip}"
            fi
            note="${note%, }"  # trim trailing comma-space
        fi

        row="| $repo_name | $mode | $embedder_name | $embedder_dim | ${file_count:--} | ${func_count:--} | ${wall_s:-0} | $status | ${note:-} |"
        echo "$row" >>"$RESULTS_FILE"
        echo "  -> $status"

    done
done

# --- Print final results table ---
echo ""
echo "=========================="
echo "  FINAL RESULTS TABLE"
echo "=========================="
cat "$RESULTS_FILE"
