from __future__ import annotations

from typing import Sequence


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def apply_smoke_limit(data: Sequence, smoke_test: bool, limit: int):
    if not smoke_test:
        return data
    return list(data[:limit])

