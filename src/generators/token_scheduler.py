import math
from typing import Optional

from src.config.schema import PromptsConfig


class TokenScheduler:
    """
    Picks the target input token size at a given elapsed time.
    Supports staged and ramp strategies while clamping to configured min/max.
    """

    def __init__(self, prompts: PromptsConfig, total_duration: float):
        self.prompts = prompts
        self.total_duration = total_duration

        # Default strategy resolution
        if prompts.stages and prompts.strategy == "uniform":
            self.strategy = "staged"
        else:
            self.strategy = prompts.strategy

    def target_tokens(self, elapsed_seconds: float) -> Optional[int]:
        """
        Returns the token count the prompt generator should aim for.
        """
        min_tok = self.prompts.min_tokens
        max_tok = self.prompts.max_tokens

        if self.strategy == "staged" and self.prompts.stages:
            return self._staged_tokens(elapsed_seconds, min_tok, max_tok)

        if self.strategy in ("linear", "exponential"):
            return self._ramp_tokens(elapsed_seconds, min_tok, max_tok)

        # Default uniform/random between min/max (return None to keep randomness)
        return None

    def _staged_tokens(self, elapsed: float, min_tok: int, max_tok: int) -> int:
        cumulative = 0.0
        for stage in self.prompts.stages:
            cumulative += stage.duration_seconds
            if elapsed <= cumulative:
                return self._clamp(stage.tokens, min_tok, max_tok)
        # If we exceeded configured stages, stick to last stage
        return self._clamp(self.prompts.stages[-1].tokens, min_tok, max_tok)

    def _ramp_tokens(self, elapsed: float, min_tok: int, max_tok: int) -> int:
        ramp = self.prompts.ramp
        if not ramp:
            start = min_tok
            end = max_tok
            duration = self.total_duration
        else:
            start = ramp.start_tokens
            end = ramp.end_tokens
            duration = ramp.duration_seconds

        progress = min(1.0, max(0.0, elapsed / max(duration, 1e-6)))
        if self.strategy == "exponential":
            value = start * math.exp(math.log(end / start) * progress)
        else:
            value = start + (end - start) * progress

        return self._clamp(int(value), min_tok, max_tok)

    @staticmethod
    def _clamp(value: int, min_tok: int, max_tok: int) -> int:
        return max(min_tok, min(value, max_tok))
