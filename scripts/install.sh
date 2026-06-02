#!/usr/bin/env bash
# ── Cairn — One-line install ─────────────────────
#
# Usage:   curl -fsSL <url>/scripts/install.sh | bash
#          bash scripts/install.sh
#
# Installs Ollama (if not present), pulls required models, installs
# the gateway package, and initializes a project.

set -euo pipefail

echo "=== Cairn Installer ==="
echo ""

# ── Ollama ───────────────────────────────────────────────────────

install_ollama() {
    if command -v ollama &>/dev/null; then
        echo "[✓] Ollama already installed ($(ollama --version 2>/dev/null || echo 'ok'))"
        return
    fi
    echo "[ ] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[✓] Ollama installed"
}

# ── Models ───────────────────────────────────────────────────────

pull_models() {
    echo "[ ] Checking Ollama models..."

    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo "[✓] nomic-embed-text already present"
    else
        echo "[ ] Pulling nomic-embed-text (~274MB)..."
        ollama pull nomic-embed-text
        echo "[✓] nomic-embed-text pulled"
    fi

    if ollama list 2>/dev/null | grep -q "qwen2.5-coder:3b"; then
        echo "[✓] qwen2.5-coder:3b already present"
    else
        echo "[ ] Pulling qwen2.5-coder:3b (~2GB)..."
        ollama pull qwen2.5-coder:3b
        echo "[✓] qwen2.5-coder:3b pulled"
    fi
}

# ── Python package ───────────────────────────────────────────────

install_gateway() {
    echo "[ ] Installing cairn package..."
    pip install --break-system-packages -e ".[dev]" 2>/dev/null \
        || pip install -e ".[dev]"
    echo "[✓] cairn installed"
}

# ── Init ─────────────────────────────────────────────────────────

init_project() {
    local target="${1:-.}"
    echo "[ ] Initializing cairn in ${target}..."

    pushd "${target}" >/dev/null

    if [ ! -f ".cairn/config.yaml" ]; then
        cairn init
    else
        echo "[✓] .cairn/config.yaml already exists"
    fi

    echo "[ ] Running quick re-index..."
    cairn reindex --mode quick || echo "[!] Re-index failed (may be OK on empty repo)"

    popd >/dev/null
    echo "[✓] Project initialized"
}

# ── Main ─────────────────────────────────────────────────────────

main() {
    install_ollama

    # Start Ollama if not running
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "[ ] Starting Ollama server..."
        ollama serve &
        sleep 3
    fi

    pull_models
    install_gateway

    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  1. Navigate to your project:  cd /path/to/project"
    echo "  2. Initialize:                 cairn init"
    echo "  3. Index:                      cairn reindex"
    echo "  4. Start:                      cairn start-all"
    echo "  5. Check health:               cairn doctor"
    echo ""

    if [ "${1:-}" != "--no-init" ]; then
        init_project "${2:-.}"
    fi
}

main "$@"
