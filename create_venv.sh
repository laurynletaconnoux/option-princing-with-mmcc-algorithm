#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
PYTHON_BIN="python3"

if [[ "$(uname -s)" == "Darwin" ]]; then
	DEFAULT_PY_ARCH="$(python3 -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
else
	DEFAULT_PY_ARCH="unknown"
fi

if [[ "$(uname -s)" == "Darwin" && "$DEFAULT_PY_ARCH" == "x86_64" ]]; then
	# Sur macOS x86_64, les wheels torch récents ne sont plus publiés pour Python >= 3.12.
	# On privilégie Python 3.11 pour garantir une installation binaire.
	if command -v pyenv >/dev/null 2>&1 && [[ -x "$(pyenv root)/versions/3.11.10/bin/python3" ]]; then
		PYTHON_BIN="$(pyenv root)/versions/3.11.10/bin/python3"
	elif command -v python3.11 >/dev/null 2>&1; then
		PYTHON_BIN="python3.11"
	else
		echo "✗ Python 3.11 introuvable. Installe Python 3.11 (ou pyenv 3.11.10) puis relance."
		exit 1
	fi
fi

if [[ -d "$VENV_DIR" ]]; then
	echo "→ Suppression de l'ancien venv: $VENV_DIR"
	rm -rf "$VENV_DIR"
fi

echo "→ Création du venv dans $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "→ Python utilisé: $("$VENV_DIR/bin/python" -V)"

echo "→ Installation des dépendances depuis $REQUIREMENTS"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo "→ Installation du package en mode éditable"
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"

echo ""
echo "✓ Venv prêt. Pour l'activer :"
echo "  source .venv/bin/activate"
