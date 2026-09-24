"""Pretty-print the JSON-RPC responses produced by test_server.sh.

Reads newline-delimited JSON from stdin, prints one summary line per message and saves
any image content blocks to mcp/output/test_server_<id>.png.
"""
import base64
import json
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

PERF_KEYS = ("thrust_N", "power_W", "efficiency", "figure_of_merit", "CP", "advance_ratio", "tip_speed_ratio")


def summarize_text(block, parts):
    try:
        t = json.loads(block["text"])
    except ValueError:
        parts.append(block["text"][:300])
        return
    if "error" in t:
        parts.append("ERROR: " + t["error"])
        return
    keys = {k: (v if not isinstance(v, (dict, list)) else f"<{type(v).__name__} len {len(v)}>") for k, v in t.items()}
    perf = t.get("performance") or t.get("optimized") or {}
    short = {k: perf[k] for k in PERF_KEYS if k in perf}
    line = json.dumps(keys)[:400]
    if short:
        line += "  performance=" + json.dumps(short)
    if "status" in t:
        line += f"  status={t['status']}"
    parts.append(line)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        print("non-JSON line on stdout (this would break MCP clients):", line[:200])
        continue
    mid = msg.get("id")
    if "error" in msg:
        print(f"id={mid}: JSON-RPC error {msg['error']}")
        continue
    result = msg.get("result", {})
    if mid == 1:
        info = result["serverInfo"]
        print(f"id=1 initialize -> server {info['name']} v{info['version']}, protocol {result['protocolVersion']}")
    elif mid == 2:
        print(f"id=2 tools/list -> {[t['name'] for t in result['tools']]}")
    elif isinstance(result, dict) and "content" in result:
        parts = []
        for block in result["content"]:
            if block.get("type") == "image":
                data = base64.b64decode(block["data"])
                path = os.path.join(OUT_DIR, f"test_server_{mid}.png")
                with open(path, "wb") as f:
                    f.write(data)
                parts.append(f"<image {block.get('mimeType')} {len(data)} bytes -> {os.path.relpath(path)}>")
            elif block.get("type") == "text":
                summarize_text(block, parts)
        print(f"id={mid}: " + " | ".join(parts))
    else:
        print(f"id={mid}: {json.dumps(result)[:300]}")
