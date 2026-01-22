from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    name: str = "default"
    base_url: str
    model_alias: Optional[str] = None
    weight: float = 1.0


class WorkloadConfig(BaseModel):
    users: int = Field(gt=0, description="Number of concurrent users")
    duration_seconds: float = Field(gt=0, description="Test duration in seconds")
    ramp_up_seconds: float = Field(default=0, ge=0, description="Ramp-up period in seconds")
    seed: Optional[int] = 42
    scenario: Optional[str] = None
    iterations: int = Field(default=1, ge=1, description="Number of consecutive runs")


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


class PrometheusConfig(BaseModel):
    enabled: bool = False
    pushgateway_url: str = "localhost:9091"
    job_name: str = "llm_load_test"
    instance_name: str = "local"
    push_interval_seconds: float = 5.0
    username: Optional[str] = None
    password: Optional[str] = None


class ClientConfig(BaseModel):
    timeout_seconds: float = 60.0
    connect_timeout: float = 10.0
    retries: int = 3
    backoff_factor: float = 1.5


class PingConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 60.0
    count: int = 4


class SystemMetricsConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 15.0
    gpu_command: str = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"


class GlobalConfig(BaseModel):
    server: Optional[ServerConfig] = None
    servers: Optional[List[ServerConfig]] = None
    workload: WorkloadConfig
    prompts: PromptsConfig
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    ping: PingConfig = Field(default_factory=PingConfig)
    system: SystemMetricsConfig = Field(default_factory=SystemMetricsConfig)
    comparison_mode: bool = False

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
