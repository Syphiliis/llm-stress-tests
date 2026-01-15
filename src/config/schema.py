from typing import List, Optional, Union
from pydantic import BaseModel, HttpUrl, Field

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

class PromptsConfig(BaseModel):
    min_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    prefix: str = ""

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

class GlobalConfig(BaseModel):
    server: Optional[ServerConfig] = None
    servers: Optional[List[ServerConfig]] = None
    workload: WorkloadConfig
    prompts: PromptsConfig
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)

    def get_servers(self) -> List[ServerConfig]:
        """Returns a list of servers, handling both single 'server' and 'servers' list."""
        if self.servers:
            return self.servers
        if self.server:
            # Normalize single server to list
            s = self.server
            return [s]
        raise ValueError("No server configuration found (neither 'server' nor 'servers')")

    @property
    def is_mixed_warfare(self) -> bool:
        return self.servers is not None and len(self.servers) > 1
