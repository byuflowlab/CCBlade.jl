#!/usr/bin/env bash
# Open the rotor explorer notebook in a browser.
#
#   ./mcp/notebook/run_notebook.sh
#   JULIA=/path/to/julia ./mcp/notebook/run_notebook.sh
#
# Pluto prints a localhost URL with a secret token; open it. The first run compiles
# CCBlade, Plots and Ipopt and takes a minute or two.
set -euo pipefail
cd "$(dirname "$0")/../.."
JULIA="${JULIA:-julia}"
exec "$JULIA" --startup-file=no --project=mcp -e '
    using Pluto
    Pluto.run(notebook = joinpath("mcp", "notebook", "rotor_explorer.jl"))'
