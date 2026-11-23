# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level detection of consecutive, end-aligned token repetitions."""

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class RepetitionDetectionParams:
    max_pattern_size: int = 0
    min_pattern_size: int = 0
    min_count: int = 0

    def __post_init__(self):
        for name in ("max_pattern_size", "min_pattern_size", "min_count"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"repetition_detection.{name} must be an integer")
        if not 0 <= self.min_pattern_size <= self.max_pattern_size:
            raise ValueError(
                "repetition_detection requires 0 <= min_pattern_size <= max_pattern_size"
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise ValueError("repetition_detection.min_count must be >= 2 when enabled")


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
    *,
    end: Optional[int] = None,
) -> bool:
    """Check a generated prefix without copying the full output history."""
    if params.max_pattern_size <= 0 or params.min_count < 2:
        return False
    if end is None:
        end = len(token_ids)
    max_size = min(params.max_pattern_size, end // params.min_count)
    for size in range(max(1, params.min_pattern_size), max_size + 1):
        for offset in range(1, size + 1):
            target = token_ids[end - offset]
            if any(
                token_ids[end - size * repeat - offset] != target
                for repeat in range(1, params.min_count)
            ):
                break
        else:
            return True
    return False
