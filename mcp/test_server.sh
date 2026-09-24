#!/usr/bin/env bash
# End-to-end check of the CCBlade MCP server over stdio, with no LLM involved.
# Sends the JSON-RPC handshake, lists the tools and calls each one, then prints a
# summary of every response. Images are saved under mcp/output/ for inspection.
#
#   ./mcp/test_server.sh                 # uses `julia` from PATH
#   JULIA=/path/to/julia ./mcp/test_server.sh
set -uo pipefail
cd "$(dirname "$0")/.."
JULIA="${JULIA:-julia}"
STDERR_LOG="$(mktemp -t ccblade-mcp-stderr)"
mkdir -p mcp/output

call() {  # id name json-arguments
  printf '{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"%s","arguments":%s}}\n' "$1" "$2" "$3"
}

{
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-server","version":"0.0.1"}}}'
  printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
  call 3  list_presets        '{}'
  call 4  analyze_rotor       '{"Vinf_m_s":5.0,"rpm":5400}'
  call 5  analyze_rotor       '{"preset":"nrel_5mw"}'
  call 6  analyze_rotor       '{"preset":"nasa_hover_rotor","pitch_deg":8,"include_sections":true}'
  call 7  sweep_rotor         '{"rpm":5400,"sweep_variable":"advance_ratio","values":[0.1,0.3,0.5,0.7]}'
  call 8  convert_rotor_units '{"convention":"propeller","Rtip_m":0.127,"rpm":5400,"Vinf_m_s":5.0,"thrust_N":3.13,"power_W":33.1}'
  call 9  plot_geometry       '{"preset":"nrel_5mw"}'
  call 10 plot_performance    '{"preset":"nrel_5mw","sweep_variable":"tip_speed_ratio","values":[3,5,7,9,11]}'
  call 11 plot_spanwise       '{"Vinf_m_s":5.0,"rpm":5400}'
  call 12 plot_airfoil        '{"airfoil":"naca4412.dat"}'
  call 13 optimize_rotor      '{"Vinf_m_s":10.0,"rpm":5400,"objective":"min_power","thrust_min_N":2.0}'
  call 14 export_blade_vtk    '{"Vinf_m_s":5.0,"rpm":5400,"name":"test_server_apc"}'
  call 15 analyze_rotor       '{"Vinf_m_s":5.0,"rpm":5400,"airfoil":"does_not_exist.dat"}'
} | "$JULIA" --startup-file=no --project=mcp mcp/server.jl 2>"$STDERR_LOG" \
  | python3 mcp/test_server_print.py
status=("${PIPESTATUS[@]}")
echo "server stderr log: $STDERR_LOG"
if [ "${status[1]}" != "0" ]; then
  echo "SERVER EXITED WITH STATUS ${status[1]}; last stderr lines:"
  grep -v '^\s*$' "$STDERR_LOG" | grep -v 'notifications/message' | tail -25
  exit 1
fi
echo "(warm-up: $(grep -o 'warm-up complete.*' "$STDERR_LOG" | head -1 || echo skipped))"
