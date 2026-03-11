from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    name: str = "default"
    base_url: str
    model_alias: Optional[str] = None
    weight: float = Field(default=1.0, gt=0)


class WorkloadConfig(BaseModel):
    users: int = Field(gt=0, description="Number of concurrent users")
    duration_seconds: float = Field(gt=0, description="Test duration in seconds")
    ramp_up_seconds: float = Field(default=0, ge=0, description="Ramp-up period in seconds")
    seed: Optional[int] = 42
    scenario: Optional[str] = None
    iterations: int = Field(default=1, ge=1, description="Number of consecutive runs")
    mode: Literal["closed_loop", "open_loop", "hybrid"] = "closed_loop"
    max_in_flight: int = Field(default=0, ge=0, description="0 means unlimited")
    arrival_distribution: Literal["deterministic", "poisson"] = "deterministic"


class TokenStageConfig(BaseModel):
    tokens: int = Field(gt=0, description="Fixed input token size for the stage")
    duration_seconds: float = Field(gt=0, description="How long to keep this size")


class TokenRampConfig(BaseModel):
    start_tokens: int = Field(gt=0)
    end_tokens: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    mode: Literal["linear", "exponential"] = "linear"


class PromptsConfig(BaseModel):
    min_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    prefix: str = ""
    # Progressive sizing strategy for input tokens
    strategy: Literal["uniform", "staged", "linear", "exponential"] = "uniform"
    stages: Optional[List[TokenStageConfig]] = None
    ramp: Optional[TokenRampConfig] = None


class LoadSegmentConfig(BaseModel):
    kind: Literal["constant", "ramp", "burst", "spike", "wave", "cooldown"]
    duration_seconds: float = Field(gt=0)
    target_users: Optional[int] = Field(default=None, ge=0)
    target_rps: Optional[float] = Field(default=None, ge=0)
    start_users: Optional[int] = Field(default=None, ge=0)
    end_users: Optional[int] = Field(default=None, ge=0)
    start_rps: Optional[float] = Field(default=None, ge=0)
    end_rps: Optional[float] = Field(default=None, ge=0)
    min_users: Optional[int] = Field(default=None, ge=0)
    max_users: Optional[int] = Field(default=None, ge=0)
    min_rps: Optional[float] = Field(default=None, ge=0)
    max_rps: Optional[float] = Field(default=None, ge=0)
    base_users: Optional[int] = Field(default=None, ge=0)
    peak_users: Optional[int] = Field(default=None, ge=0)
    base_rps: Optional[float] = Field(default=None, ge=0)
    peak_rps: Optional[float] = Field(default=None, ge=0)
    every_seconds: Optional[float] = Field(default=None, gt=0)
    burst_duration_seconds: Optional[float] = Field(default=None, gt=0)
    start_at_seconds: float = Field(default=0, ge=0)
    hold_seconds: Optional[float] = Field(default=None, gt=0)
    period_seconds: Optional[float] = Field(default=None, gt=0)
    shape: Literal["sine", "triangle", "sawtooth"] = "sine"

    @model_validator(mode="after")
    def validate_profile_shape(self) -> "LoadSegmentConfig":
        if self.kind == "constant":
            return self

        if self.kind in ("ramp", "cooldown"):
            has_users = self.start_users is not None and self.end_users is not None
            has_rps = self.start_rps is not None and self.end_rps is not None
            if not has_users and not has_rps:
                raise ValueError(f"{self.kind} segments require start/end users or start/end rps.")

        if self.kind == "burst":
            has_users = self.base_users is not None and self.peak_users is not None
            has_rps = self.base_rps is not None and self.peak_rps is not None
            if not has_users and not has_rps:
                raise ValueError("burst segments require base/peak users or base/peak rps.")
            if self.every_seconds is None or self.burst_duration_seconds is None:
                raise ValueError("burst segments require every_seconds and burst_duration_seconds.")

        if self.kind == "spike":
            has_users = self.base_users is not None and self.peak_users is not None
            has_rps = self.base_rps is not None and self.peak_rps is not None
            if not has_users and not has_rps:
                raise ValueError("spike segments require base/peak users or base/peak rps.")
            if self.hold_seconds is None:
                raise ValueError("spike segments require hold_seconds.")

        if self.kind == "wave":
            has_users = self.min_users is not None and self.max_users is not None
            has_rps = self.min_rps is not None and self.max_rps is not None
            if not has_users and not has_rps:
                raise ValueError("wave segments require min/max users or min/max rps.")
            if self.period_seconds is None:
                raise ValueError("wave segments require period_seconds.")

        return self


class LoadProfileConfig(BaseModel):
    type: Literal["constant", "ramp", "burst", "spike", "wave", "cooldown", "composite"] = "constant"
    target_users: Optional[int] = Field(default=None, ge=0)
    target_rps: Optional[float] = Field(default=None, ge=0)
    start_users: Optional[int] = Field(default=None, ge=0)
    end_users: Optional[int] = Field(default=None, ge=0)
    start_rps: Optional[float] = Field(default=None, ge=0)
    end_rps: Optional[float] = Field(default=None, ge=0)
    min_users: Optional[int] = Field(default=None, ge=0)
    max_users: Optional[int] = Field(default=None, ge=0)
    min_rps: Optional[float] = Field(default=None, ge=0)
    max_rps: Optional[float] = Field(default=None, ge=0)
    base_users: Optional[int] = Field(default=None, ge=0)
    peak_users: Optional[int] = Field(default=None, ge=0)
    base_rps: Optional[float] = Field(default=None, ge=0)
    peak_rps: Optional[float] = Field(default=None, ge=0)
    every_seconds: Optional[float] = Field(default=None, gt=0)
    burst_duration_seconds: Optional[float] = Field(default=None, gt=0)
    start_at_seconds: float = Field(default=0, ge=0)
    hold_seconds: Optional[float] = Field(default=None, gt=0)
    period_seconds: Optional[float] = Field(default=None, gt=0)
    shape: Literal["sine", "triangle", "sawtooth"] = "sine"
    segments: Optional[List[LoadSegmentConfig]] = None

    @model_validator(mode="after")
    def validate_profile(self) -> "LoadProfileConfig":
        if self.type == "composite":
            if not self.segments:
                raise ValueError("Composite load profiles require at least one segment.")
            return self

        # Reuse the segment validator for single-profile presets.
        LoadSegmentConfig(
            kind=self.type,
            duration_seconds=1.0,
            target_users=self.target_users,
            target_rps=self.target_rps,
            start_users=self.start_users,
            end_users=self.end_users,
            start_rps=self.start_rps,
            end_rps=self.end_rps,
            min_users=self.min_users,
            max_users=self.max_users,
            min_rps=self.min_rps,
            max_rps=self.max_rps,
            base_users=self.base_users,
            peak_users=self.peak_users,
            base_rps=self.base_rps,
            peak_rps=self.peak_rps,
            every_seconds=self.every_seconds,
            burst_duration_seconds=self.burst_duration_seconds,
            start_at_seconds=self.start_at_seconds,
            hold_seconds=self.hold_seconds,
            period_seconds=self.period_seconds,
            shape=self.shape,
        )
        return self


class ThinkTimeConfig(BaseModel):
    enabled: bool = False
    distribution: Literal["none", "fixed", "uniform", "exponential", "lognormal"] = "none"
    fixed_seconds: Optional[float] = Field(default=None, ge=0)
    min_seconds: float = Field(default=0.0, ge=0)
    max_seconds: Optional[float] = Field(default=None, ge=0)
    mean_seconds: Optional[float] = Field(default=None, gt=0)
    sigma: float = Field(default=0.75, gt=0)
    jitter_ratio: float = Field(default=0.0, ge=0, le=1)
    after_error_multiplier: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_distribution(self) -> "ThinkTimeConfig":
        if not self.enabled or self.distribution == "none":
            return self

        if self.distribution == "fixed" and self.fixed_seconds is None and self.mean_seconds is None:
            raise ValueError("fixed think_time requires fixed_seconds or mean_seconds.")

        if self.distribution == "uniform" and self.max_seconds is None:
            raise ValueError("uniform think_time requires max_seconds.")

        if self.distribution in ("exponential", "lognormal") and self.mean_seconds is None:
            raise ValueError(f"{self.distribution} think_time requires mean_seconds.")

        if self.max_seconds is not None and self.max_seconds < self.min_seconds:
            raise ValueError("think_time.max_seconds must be greater than or equal to min_seconds.")

        return self


class PrometheusConfig(BaseModel):
    enabled: bool = False
    pushgateway_url: str = "localhost:9091"
    job_name: str = "llm_load_test"
    instance_name: str = "local"
    push_interval_seconds: float = 5.0
    username: Optional[str] = None
    password: Optional[str] = None


class ClientConfig(BaseModel):
    timeout_seconds: float = Field(default=60.0, gt=0)
    connect_timeout: float = Field(default=10.0, gt=0)
    retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=1.5, gt=0)


class PingConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 60.0
    count: int = 4


class SystemMetricsConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 15.0
    gpu_command: str = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    source: Literal["local", "ssh"] = "local"
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_options: List[str] = Field(default_factory=lambda: ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"])
    include_local_cpu_ram: bool = False

    @model_validator(mode="after")
    def validate_remote_source(self) -> "SystemMetricsConfig":
        if self.source == "ssh" and not self.ssh_host:
            raise ValueError("system.source='ssh' requires system.ssh_host.")
        return self


class GlobalConfig(BaseModel):
    server: Optional[ServerConfig] = None
    servers: Optional[List[ServerConfig]] = None
    workload: WorkloadConfig
    prompts: PromptsConfig
    load_profile: Optional[LoadProfileConfig] = None
    think_time: ThinkTimeConfig = Field(default_factory=ThinkTimeConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    ping: PingConfig = Field(default_factory=PingConfig)
    system: SystemMetricsConfig = Field(default_factory=SystemMetricsConfig)
    comparison_mode: bool = False

    @model_validator(mode="after")
    def validate_topology(self) -> "GlobalConfig":
        has_single_server = self.server is not None
        has_multiple_servers = bool(self.servers)

        if has_single_server == has_multiple_servers:
            raise ValueError("Configure exactly one of 'server' or 'servers'.")

        if self.comparison_mode and (not self.servers or len(self.servers) < 2):
            raise ValueError("comparison_mode requires at least two entries in 'servers'.")

        return self

    def get_servers(self) -> List[ServerConfig]:
        """Returns a list of servers, handling both single 'server' and 'servers' list."""
        if self.servers:
            return self.servers
        if self.server:
            return [self.server]
        raise ValueError("No server configuration found (neither 'server' nor 'servers')")

    @property
    def is_mixed_warfare(self) -> bool:
        return self.servers is not None and len(self.servers) > 1
