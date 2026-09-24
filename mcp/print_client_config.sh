#!/usr/bin/env bash
# Print the client registration command and config for THIS machine, with absolute
# paths filled in. Nothing is modified; copy what you need.
#
#   ./mcp/print_client_config.sh
#   JULIA=/path/to/julia ./mcp/print_client_config.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
JULIA="${JULIA:-julia}"
# Resolve to a real path: MCP clients do not read your shell aliases or PATH.
JULIA_BIN="$(command -v "$JULIA" 2>/dev/null)"
if [ -z "$JULIA_BIN" ]; then
  echo "Julia not found as '$JULIA'. Re-run as: JULIA=/path/to/julia $0" >&2
  exit 1
fi

cat <<TXT
Julia:     $JULIA_BIN
Project:   $ROOT/mcp

--- Claude Code (run this) ---

claude mcp add ccblade -- $JULIA_BIN --startup-file=no --project=$ROOT/mcp $ROOT/mcp/server.jl

--- Claude Desktop (Settings > Developer > Edit Config, merge this in, then restart) ---

{
  "mcpServers": {
    "ccblade": {
      "command": "$JULIA_BIN",
      "args": [
        "--startup-file=no",
        "--project=$ROOT/mcp",
        "$ROOT/mcp/server.jl"
      ]
    }
  }
}

--- MCP Inspector (shows every request and response in a browser) ---

npx @modelcontextprotocol/inspector $JULIA_BIN --startup-file=no --project=$ROOT/mcp $ROOT/mcp/server.jl
TXT
