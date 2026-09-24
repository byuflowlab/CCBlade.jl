#!/usr/bin/env bash
# Run every check before a demo: library values, notebook cell logic, a headless Pluto
# run, and the whole MCP server over raw JSON-RPC. Takes a couple of minutes.
#
#   ./mcp/check_all.sh
#   JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia ./mcp/check_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
JULIA="${JULIA:-julia}"
fails=0

step() {  # label command...
  local label="$1"; shift
  printf '\n=== %s ===\n' "$label"
  if "$@"; then
    printf '  OK: %s\n' "$label"
  else
    printf '  FAILED: %s\n' "$label"
    fails=$((fails + 1))
  fi
}

step "library values vs CCBlade docs" \
  "$JULIA" --startup-file=no --project=mcp mcp/test/run_lib_tests.jl
step "notebook cell logic" \
  "$JULIA" --startup-file=no --project=mcp mcp/test/run_notebook_tests.jl
step "notebook runs in Pluto (all branches)" \
  "$JULIA" --startup-file=no --project=mcp mcp/test/check_notebook_runs.jl --buttons
step "MCP server over stdio" ./mcp/smoke_test.sh

printf '\n'
if [ "$fails" -eq 0 ]; then
  echo "ALL CHECKS PASSED - ready to demo."
else
  echo "$fails CHECK(S) FAILED - see above."
fi
exit "$fails"
