"""
Tail MetaField field-automata events (echo_events.jsonl).

Non-blocking: only reads new lines each poll.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class AutomataEventTail:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._pos = 0
        self.last_gates: List[str] = []
        self.last_event: Optional[Dict[str, Any]] = None
        self.n_events = 0
        if self.path.exists():
            # start at end — only live events
            self._pos = self.path.stat().st_size
        print(f"[automata] tailing events → {self.path}")

    def poll(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._pos:
            self._pos = 0  # truncated
        if size == self._pos:
            return []

        out: List[Dict[str, Any]] = []
        try:
            with self.path.open(encoding="utf-8") as fh:
                fh.seek(self._pos)
                for line in fh:
                    text = line.strip()
                    if not text.startswith("{"):
                        continue
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("kind") != "field_automata_event":
                        continue
                    out.append(obj)
                    self.last_event = obj
                    self.last_gates = list(obj.get("gates") or [])
                    self.n_events += 1
                self._pos = fh.tell()
        except OSError:
            return []
        return out
