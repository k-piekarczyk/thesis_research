from collections import deque
from typing import Any, List


class FixedList:

    internal = deque()

    def __init__(self, length) -> None:
        self.length = length
    
    def push(self, val: Any) -> None:
        if len(self.internal) < self.length:
            self.internal.append(val)
        elif len(self.internal) == self.length:
            self.internal.rotate(-1)
            self.internal[-1] = val
        else:
            raise Exception("Internal deque somehow")
    
    def get(self) -> List[Any]:
        return self.internal