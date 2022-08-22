from typing import Protocol


class Scorer(Protocol):
    def process(self, *args, **kwargs):
        raise NotImplementedError