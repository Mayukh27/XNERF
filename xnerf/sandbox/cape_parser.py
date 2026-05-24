from __future__ import annotations

import json
from pathlib import Path


def parse_cape_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    behavior = report.get("behavior", {})
    api_calls = []
    for proc in behavior.get("processes", []):
        for call in proc.get("calls", []):
            api_calls.append(call.get("api", "UNK"))
    network = report.get("network", {})
    events = []
    for key in ("dns", "http", "tcp", "udp"):
        for item in network.get(key, []):
            events.append({"type": key, "value": item})
    return {"api_calls": api_calls, "network_events": events}

