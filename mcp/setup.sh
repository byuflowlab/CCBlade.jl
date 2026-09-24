#!/usr/bin/env bash
# One-time setup for the CCBlade MCP workshop.
#
#   ./mcp/setup.sh
#   JULIA=/path/to/julia ./mcp/setup.sh
#
# Installs the exact package versions in mcp/Manifest.toml, precompiles them, and runs a
# quick self-check. Takes 5-15 minutes on a first run, almost all of it precompilation.
set -uo pipefail
cd "$(dirname "$0")/.."
JULIA="${JULIA:-julia}"

if ! command -v "$JULIA" >/dev/null 2>&1; then
  echo "Julia not found as '$JULIA'."
  echo "Install Julia 1.10 or newer from https://julialang.org/downloads (or 'juliaup add release'),"
  echo "then re-run, optionally as: JULIA=/path/to/julia ./mcp/setup.sh"
  exit 1
fi

version=$("$JULIA" --startup-file=no -e 'print(VERSION)')
echo "Using $JULIA (Julia $version)"
case "$version" in
  1.[0-9].*) echo "WARNING: Julia 1.10 or newer is required; this is $version." ;;
esac

echo
echo "Installing packages (this is the slow part; grab a coffee)..."
"$JULIA" --startup-file=no --project=mcp -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()' || { echo "FAILED: package installation"; exit 1; }

echo
echo "Running a quick self-check..."
"$JULIA" --startup-file=no --project=mcp mcp/test/run_lib_tests.jl || {
  echo "FAILED: the library self-check did not pass."; exit 1; }

cat <<'DONE'

Setup complete.

Next:
  1. Register the server with your MCP client. Print the exact command for this
     machine with:
         ./mcp/print_client_config.sh
  2. Or open the interactive notebook (no LLM needed):
         ./mcp/notebook/run_notebook.sh

Full instructions: mcp/README.md
DONE
