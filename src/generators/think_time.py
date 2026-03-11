import math
import random

from src.config.schema import ThinkTimeConfig


class ThinkTimeSampler:
    """
    Generate user think time pauses between requests.
    """

    def __init__(self, config: ThinkTimeConfig, seed: int = 42):
        self.config = config
        self._rng = random.Random(seed)

    def sample(self, after_error: bool = False) -> float:
        if not self.config.enabled or self.config.distribution == "none":
            return 0.0

        if self.config.distribution == "fixed":
            value = self.config.fixed_seconds
            if value is None:
                value = self.config.mean_seconds or self.config.min_seconds
        elif self.config.distribution == "uniform":
            upper = self.config.max_seconds if self.config.max_seconds is not None else self.config.min_seconds
            value = self._rng.uniform(self.config.min_seconds, upper)
        elif self.config.distribution == "exponential":
            value = self._rng.expovariate(1.0 / self.config.mean_seconds)
        else:
            sigma = self.config.sigma
            mean_seconds = self.config.mean_seconds or max(self.config.min_seconds, 1e-6)
            mu = math.log(max(mean_seconds, 1e-6)) - ((sigma ** 2) / 2.0)
            value = self._rng.lognormvariate(mu, sigma)

        if after_error and self.config.after_error_multiplier != 1.0:
            value *= self.config.after_error_multiplier

        if self.config.jitter_ratio > 0:
            low = max(0.0, 1.0 - self.config.jitter_ratio)
            high = 1.0 + self.config.jitter_ratio
            value *= self._rng.uniform(low, high)

        if value < self.config.min_seconds:
            value = self.config.min_seconds

        if self.config.max_seconds is not None:
            value = min(value, self.config.max_seconds)

        return value
