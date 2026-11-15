"""自然文件名排序：page2 排在 page10 之前，而不是字典序。"""

import re

_PART = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    parts = []
    for piece in _PART.split(name):
        if piece.isdigit():
            parts.append((0, int(piece), ""))
        elif piece:
            parts.append((1, 0, piece.casefold()))
    return tuple(parts)


def natural_sorted(names: list[str]) -> list[str]:
    return sorted(names, key=natural_key)