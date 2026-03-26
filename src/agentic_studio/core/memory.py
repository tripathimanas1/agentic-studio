from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from agentic_studio.core.models import AgentMessage


@dataclass
class WorkingMemory:
    max_items: int = 40
    _messages: deque[AgentMessage] = field(default_factory=deque)

    def add(self, role: str, content: str) -> None:
        self._messages.append(AgentMessage(role=role, content=content))
        while len(self._messages) > self.max_items:
            self._messages.popleft()

    def recent(self, n: int = 8) -> list[AgentMessage]:
        return list(self._messages)[-n:]

    def render(self, n: int = 8) -> str:
        msgs = self.recent(n)
        return "\n".join(f"[{m.role}] {m.content}" for m in msgs)