from typing import Hashable

from kernels.hasher import Hasher


class LabelHasher(Hasher):
    """Returns a simple incremental count-based hash."""

    def __init__(self, start_hash: int = 0):
        self.count = start_hash
        self.labels: dict[Hashable, int] = {}


    def hash(self, val: Hashable) -> int:
        if val not in self.labels:
            self.count += 1
            self.labels[val] = self.count
        return self.labels[val]
