from abc import ABC, abstractmethod
from typing import Hashable


class Hasher(ABC):

    @abstractmethod
    def hash(self, val: Hashable) -> int: ...
