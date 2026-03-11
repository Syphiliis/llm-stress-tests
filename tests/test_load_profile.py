import unittest

from src.config.schema import LoadProfileConfig, LoadSegmentConfig, WorkloadConfig
from src.engine.load_profile import LoadProfileScheduler


class LoadProfileSchedulerTests(unittest.TestCase):
    def test_legacy_ramp_up_is_translated_into_dynamic_users(self) -> None:
        workload = WorkloadConfig(users=10, duration_seconds=60, ramp_up_seconds=20)
        scheduler = LoadProfileScheduler(None, workload)

        self.assertEqual(scheduler.target_at(0).target_users, 1)
        self.assertEqual(scheduler.target_at(10).target_users, 6)
        self.assertEqual(scheduler.target_at(25).target_users, 10)

    def test_burst_profile_switches_between_base_and_peak(self) -> None:
        workload = WorkloadConfig(users=5, duration_seconds=60)
        profile = LoadProfileConfig(
            type="burst",
            base_users=5,
            peak_users=20,
            every_seconds=10,
            burst_duration_seconds=3,
        )
        scheduler = LoadProfileScheduler(profile, workload)

        self.assertEqual(scheduler.target_at(1).target_users, 20)
        self.assertEqual(scheduler.target_at(5).target_users, 5)
        self.assertEqual(scheduler.target_at(11).target_users, 20)

    def test_wave_profile_oscillates_between_bounds(self) -> None:
        workload = WorkloadConfig(users=5, duration_seconds=60)
        profile = LoadProfileConfig(
            type="wave",
            min_users=10,
            max_users=30,
            period_seconds=20,
            shape="triangle",
        )
        scheduler = LoadProfileScheduler(profile, workload)

        self.assertEqual(scheduler.target_at(0).target_users, 10)
        self.assertEqual(scheduler.target_at(10).target_users, 30)
        self.assertEqual(scheduler.target_at(20).target_users, 10)

    def test_composite_profile_honors_sequential_segments(self) -> None:
        workload = WorkloadConfig(users=5, duration_seconds=30)
        profile = LoadProfileConfig(
            type="composite",
            segments=[
                LoadSegmentConfig(kind="constant", duration_seconds=10, target_users=4),
                LoadSegmentConfig(kind="cooldown", duration_seconds=10, start_users=10, end_users=2),
            ],
        )
        scheduler = LoadProfileScheduler(profile, workload)

        self.assertEqual(scheduler.target_at(5).target_users, 4)
        self.assertEqual(scheduler.target_at(10).target_users, 10)
        self.assertEqual(scheduler.target_at(20).target_users, 2)

    def test_open_loop_can_report_peak_rps(self) -> None:
        workload = WorkloadConfig(users=2, duration_seconds=60, mode="open_loop")
        profile = LoadProfileConfig(type="spike", base_rps=2, peak_rps=12, start_at_seconds=5, hold_seconds=3)
        scheduler = LoadProfileScheduler(profile, workload)

        self.assertEqual(scheduler.max_target_rps(), 12)
        self.assertEqual(scheduler.target_at(2).target_rps, 2)
        self.assertEqual(scheduler.target_at(6).target_rps, 12)


if __name__ == "__main__":
    unittest.main()
