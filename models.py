from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Message:
    role: str
    content: Optional[str] = None

    def render(self):
        result = self.role + ":"
        if self.content is not None:
            result += " " + self.content
        return result