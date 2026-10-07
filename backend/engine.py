"""Deterministic mock infill with exact anchors and a per-session token cache.

This exercises the editor contract; it is not a diffusion or language model.
"""

from difflib import SequenceMatcher
from hashlib import blake2s
import re

PHRASES = (
    "vividly imagined ", "carefully composed ", "richly detailed ",
    "subtly expressive ", "atmospheric ", "striking ",
)


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def python_cursor(text: str, cursor: int) -> int:
    """Translate a DOM cursor offset, including emoji/supplementary characters."""
    return len(text.encode("utf-16-le")[:cursor * 2].decode("utf-16-le", errors="ignore"))


class MockEngine:
    def __init__(self):
        self.words = []
        self.additions = []

    def update(self, text: str, cursor: int, radius: int = 3, commit: bool = False) -> dict:
        matches = list(re.finditer(r"\S+", text))
        words = [match.group() for match in matches]
        position = python_cursor(text, cursor)
        active = min(range(len(matches)), key=lambda i: (
            0 if matches[i].start() <= position <= matches[i].end()
            else min(abs(position - matches[i].start()), abs(position - matches[i].end()))
        )) if matches else 0

        # Preserve shared ends explicitly, including long repetitive prompts
        # where SequenceMatcher's popularity heuristic omits otherwise stable words.
        prefix = 0
        limit = min(len(self.words), len(words))
        while prefix < limit and self.words[prefix] == words[prefix]:
            prefix += 1
        suffix = 0
        while suffix < limit - prefix and self.words[-suffix - 1] == words[-suffix - 1]:
            suffix += 1
        previous = {i: self.additions[i] for i in range(prefix)}
        for offset in range(1, suffix + 1):
            previous[len(words) - offset] = self.additions[-offset]
        old_end, new_end = len(self.words) - suffix, len(words) - suffix
        for block in SequenceMatcher(None, self.words[prefix:old_end], words[prefix:new_end]).get_matching_blocks():
            for offset in range(block.size):
                previous[prefix + block.b + offset] = self.additions[prefix + block.a + offset]

        segments, additions = [], []
        last_end = 0
        for i, match in enumerate(matches):
            if match.start() > last_end:
                segments.append({"text": text[last_end:match.start()], "kind": "anchor"})
            frozen = abs(i - active) > radius
            if commit:
                addition = ""
            elif i in previous and frozen:
                addition = previous[i]
            elif any(char.isalpha() for char in match.group()):
                # Sparse, reproducible infill conditioned on nearby words. Editing
                # outside the radius cannot change a cached addition.
                context = " ".join(words[max(0, i - 1):i + 2])
                digest = blake2s(context.encode("utf-8"), digest_size=2).digest()
                addition = PHRASES[digest[0] % len(PHRASES)] if i % 3 == 0 else ""
            else:
                addition = ""
            additions.append(addition)
            if addition:
                segments.append({"text": addition, "kind": "generated", "frozen": frozen})
            segments.append({"text": match.group(), "kind": "anchor"})
            last_end = match.end()
        if last_end < len(text):
            segments.append({"text": text[last_end:], "kind": "anchor"})
        self.words, self.additions = words, additions
        return {"output": "".join(segment["text"] for segment in segments), "segments": segments}
