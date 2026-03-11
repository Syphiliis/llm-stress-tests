import unittest

from src.config.schema import ThinkTimeConfig
from src.generators.think_time import ThinkTimeSampler


class ThinkTimeSamplerTests(unittest.TestCase):
    def test_fixed_distribution_returns_exact_value(self) -> None:
        sampler = ThinkTimeSampler(
            ThinkTimeConfig(enabled=True, distribution="fixed", fixed_seconds=1.5),
            seed=123,
        )

        self.assertEqual(sampler.sample(), 1.5)

    def test_uniform_distribution_stays_within_bounds(self) -> None:
        sampler = ThinkTimeSampler(
            ThinkTimeConfig(enabled=True, distribution="uniform", min_seconds=1.0, max_seconds=2.0),
            seed=123,
        )

        samples = [sampler.sample() for _ in range(20)]

        self.assertTrue(all(1.0 <= sample <= 2.0 for sample in samples))

    def test_lognormal_distribution_respects_maximum_and_error_multiplier(self) -> None:
        sampler = ThinkTimeSampler(
            ThinkTimeConfig(
                enabled=True,
                distribution="lognormal",
                mean_seconds=1.0,
                max_seconds=3.0,
                after_error_multiplier=2.0,
            ),
            seed=123,
        )

        normal_sample = sampler.sample()
        error_sample = sampler.sample(after_error=True)

        self.assertGreaterEqual(normal_sample, 0.0)
        self.assertLessEqual(normal_sample, 3.0)
        self.assertGreaterEqual(error_sample, normal_sample)
        self.assertLessEqual(error_sample, 3.0)


if __name__ == "__main__":
    unittest.main()
