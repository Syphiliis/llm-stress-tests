import math
from dataclasses import dataclass
from typing import List, Optional

from src.config.schema import LoadProfileConfig, LoadSegmentConfig, WorkloadConfig


@dataclass(frozen=True)
class LoadTarget:
    target_users: Optional[int] = None
    target_rps: Optional[float] = None
    segment_kind: str = "constant"


class LoadProfileScheduler:
    """
    Resolve user and/or RPS targets over time.

    Profiles can be primitive presets or a sequential composite of segments.
    """

    def __init__(self, profile: Optional[LoadProfileConfig], workload: WorkloadConfig):
        self.profile = profile
        self.workload = workload
        self.segments = self._build_segments()

    def target_at(self, elapsed_seconds: float) -> LoadTarget:
        if not self.segments:
            return LoadTarget(target_users=self.workload.users, segment_kind="constant")

        elapsed = max(0.0, elapsed_seconds)
        remaining = elapsed
        for index, segment in enumerate(self.segments):
            if remaining < segment.duration_seconds or index == len(self.segments) - 1:
                return self._evaluate_segment(segment, remaining)
            remaining -= segment.duration_seconds

        return self._evaluate_segment(self.segments[-1], self.segments[-1].duration_seconds)

    def max_target_users(self) -> int:
        candidates = [self.workload.users]
        for segment in self.segments:
            values = [
                segment.target_users,
                segment.start_users,
                segment.end_users,
                segment.min_users,
                segment.max_users,
                segment.base_users,
                segment.peak_users,
            ]
            candidates.extend(v for v in values if v is not None)
        return max(candidates) if candidates else self.workload.users

    def max_target_rps(self, default: float = 0.0) -> float:
        candidates = [default]
        for segment in self.segments:
            values = [
                segment.target_rps,
                segment.start_rps,
                segment.end_rps,
                segment.min_rps,
                segment.max_rps,
                segment.base_rps,
                segment.peak_rps,
            ]
            candidates.extend(v for v in values if v is not None)
        return max(candidates) if candidates else default

    def _build_segments(self) -> List[LoadSegmentConfig]:
        if self.profile:
            if self.profile.type == "composite":
                return list(self.profile.segments or [])

            return [
                LoadSegmentConfig(
                    kind=self.profile.type,
                    duration_seconds=self.workload.duration_seconds,
                    target_users=self.profile.target_users,
                    target_rps=self.profile.target_rps,
                    start_users=self.profile.start_users,
                    end_users=self.profile.end_users,
                    start_rps=self.profile.start_rps,
                    end_rps=self.profile.end_rps,
                    min_users=self.profile.min_users,
                    max_users=self.profile.max_users,
                    min_rps=self.profile.min_rps,
                    max_rps=self.profile.max_rps,
                    base_users=self.profile.base_users,
                    peak_users=self.profile.peak_users,
                    base_rps=self.profile.base_rps,
                    peak_rps=self.profile.peak_rps,
                    every_seconds=self.profile.every_seconds,
                    burst_duration_seconds=self.profile.burst_duration_seconds,
                    start_at_seconds=self.profile.start_at_seconds,
                    hold_seconds=self.profile.hold_seconds,
                    period_seconds=self.profile.period_seconds,
                    shape=self.profile.shape,
                )
            ]

        if self.workload.ramp_up_seconds > 0:
            ramp_duration = min(self.workload.ramp_up_seconds, self.workload.duration_seconds)
            segments = [
                LoadSegmentConfig(
                    kind="ramp",
                    duration_seconds=ramp_duration,
                    start_users=1,
                    end_users=self.workload.users,
                )
            ]
            remaining = self.workload.duration_seconds - ramp_duration
            if remaining > 0:
                segments.append(
                    LoadSegmentConfig(
                        kind="constant",
                        duration_seconds=remaining,
                        target_users=self.workload.users,
                    )
                )
            return segments

        return [
            LoadSegmentConfig(
                kind="constant",
                duration_seconds=self.workload.duration_seconds,
                target_users=self.workload.users,
            )
        ]

    def _evaluate_segment(self, segment: LoadSegmentConfig, elapsed: float) -> LoadTarget:
        if segment.kind == "constant":
            return LoadTarget(
                target_users=self._choose_default(segment.target_users, self.workload.users),
                target_rps=segment.target_rps,
                segment_kind=segment.kind,
            )

        if segment.kind in ("ramp", "cooldown"):
            progress = min(1.0, max(0.0, elapsed / max(segment.duration_seconds, 1e-6)))
            return LoadTarget(
                target_users=self._interpolate_int(segment.start_users, segment.end_users, progress),
                target_rps=self._interpolate_float(segment.start_rps, segment.end_rps, progress),
                segment_kind=segment.kind,
            )

        if segment.kind == "burst":
            phase = elapsed % segment.every_seconds
            is_peak = phase < segment.burst_duration_seconds
            return LoadTarget(
                target_users=segment.peak_users if is_peak else self._choose_default(segment.base_users, self.workload.users),
                target_rps=segment.peak_rps if is_peak else segment.base_rps,
                segment_kind=segment.kind,
            )

        if segment.kind == "spike":
            in_spike = segment.start_at_seconds <= elapsed < (segment.start_at_seconds + segment.hold_seconds)
            return LoadTarget(
                target_users=segment.peak_users if in_spike else self._choose_default(segment.base_users, self.workload.users),
                target_rps=segment.peak_rps if in_spike else segment.base_rps,
                segment_kind=segment.kind,
            )

        if segment.kind == "wave":
            phase = (elapsed % segment.period_seconds) / segment.period_seconds
            amplitude = self._shape_amplitude(segment.shape, phase)
            return LoadTarget(
                target_users=self._interpolate_int(segment.min_users, segment.max_users, amplitude),
                target_rps=self._interpolate_float(segment.min_rps, segment.max_rps, amplitude),
                segment_kind=segment.kind,
            )

        return LoadTarget(target_users=self.workload.users, segment_kind="constant")

    @staticmethod
    def _shape_amplitude(shape: str, phase: float) -> float:
        if shape == "triangle":
            return 1.0 - abs(2.0 * phase - 1.0)
        if shape == "sawtooth":
            return phase
        return 0.5 * (1.0 + math.sin((2.0 * math.pi * phase) - (math.pi / 2.0)))

    @staticmethod
    def _choose_default(value: Optional[int], default: int) -> int:
        return value if value is not None else default

    @staticmethod
    def _interpolate_int(start: Optional[int], end: Optional[int], progress: float) -> Optional[int]:
        if start is None and end is None:
            return None
        if start is None:
            return end
        if end is None:
            return start
        return int(round(start + (end - start) * progress))

    @staticmethod
    def _interpolate_float(start: Optional[float], end: Optional[float], progress: float) -> Optional[float]:
        if start is None and end is None:
            return None
        if start is None:
            return end
        if end is None:
            return start
        return start + (end - start) * progress
