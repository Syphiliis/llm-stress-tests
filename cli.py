#!/usr/bin/env python3
"""
Interactive CLI for LLM Stress Testing
Allows dynamic configuration of VPS IPs and test selection
"""

import asyncio
import os
import sys
import socket
import yaml
import tempfile
from datetime import datetime as dt
import csv
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Available test configurations
TEST_CONFIGS = {
    "1": {
        "name": "Flash Only",
        "file": "config/flash_only.yaml",
        "description": "Heavy load on Flash model only (faster, lighter model)",
        "single_server": True,
        "port": 38703
    },
    "2": {
        "name": "Thinker Only",
        "file": "config/thinker_only.yaml",
        "description": "Heavy load on Thinker model only (slower, more powerful model)",
        "single_server": True,
        "port": 38704
    },
    "3": {
        "name": "Dual Warfare",
        "file": "config/dual_warfare.yaml",
        "description": "Test Flash and Thinker simultaneously (50/50 split)",
        "single_server": False,
        "ports": [38703, 38704]
    },
    "4": {
        "name": "Mixed Warfare",
        "file": "config/mixed_warfare.yaml",
        "description": "Mixed load with weighted distribution (30/70 split)",
        "single_server": False,
        "ports": [8080, 8081]
    },
    "5": {
        "name": "Custom Config",
        "file": None,
        "description": "Specify your own config file",
        "single_server": None
    },
    "6": {
        "name": "Flash vs Thinker (Comparison)",
        "file": "config/comparison_flash_thinker.yaml",
        "description": "Sequential comparison using identical scenarios",
        "single_server": False,
        "ports": [38703, 38704]
    },
    "7": {
        "name": "Full Test (Flash then Thinker)",
        "file": "config/comparison_flash_thinker.yaml",
        "description": "Complete run: Flash first, then Thinker (same scenario)",
        "single_server": False,
        "ports": [38703, 38704]
    }
}


def print_header():
    """Print CLI header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}  LLM Stress Test - Interactive CLI")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


def print_section(title):
    """Print section header"""
    print(f"\n{Fore.YELLOW}▶ {title}{Style.RESET_ALL}")


def get_input(prompt, default=None):
    """Get user input with optional default value"""
    if default:
        full_prompt = f"{prompt} [{Fore.GREEN}{default}{Style.RESET_ALL}]: "
    else:
        full_prompt = f"{prompt}: "
    
    value = input(full_prompt).strip()
    return value if value else default


def validate_ip(ip):
    """Validate IP address format"""
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


def check_connectivity(host, port, timeout=2):
    """Check if a host:port is reachable"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def display_test_menu():
    """Display available test configurations"""
    print_section("Available Test Configurations")
    for key, config in TEST_CONFIGS.items():
        print(f"  {Fore.CYAN}{key}{Style.RESET_ALL}. {Fore.WHITE}{config['name']}{Style.RESET_ALL}")
        print(f"     {Fore.LIGHTBLACK_EX}{config['description']}{Style.RESET_ALL}")


def get_gpu_ip():
    """Get GPU server IP from user"""
    print_section("GPU Server Configuration")
    
    # Try to read default from existing config
    default_ip = "24.124.32.70"
    try:
        with open("config/remote_gpu.yaml", "r") as f:
            config = yaml.safe_load(f)
            if "server" in config and "base_url" in config["server"]:
                url = config["server"]["base_url"]
                # Extract IP from URL
                import re
                match = re.search(r'http://([^:]+):', url)
                if match:
                    default_ip = match.group(1)
    except Exception:
        pass
    
    while True:
        gpu_ip = get_input("Enter GPU server IP address", default_ip)
        if validate_ip(gpu_ip):
            return gpu_ip
        else:
            print(f"{Fore.RED}✗ Invalid IP address. Please try again.{Style.RESET_ALL}")


def get_custom_ports(test_choice):
    """Ask user if they want to customize ports"""
    config_info = TEST_CONFIGS[test_choice]
    
    if test_choice == "5":  # Custom config
        return None
    
    print_section("Port Configuration")
    
    if config_info["single_server"]:
        default_port = config_info["port"]
        print(f"  Default port: {default_port}")
        customize = get_input("Use custom port? (y/n)", "n").lower()
        
        if customize == "y":
            while True:
                try:
                    port = int(get_input("Enter port number", str(default_port)))
                    if 1 <= port <= 65535:
                        return port
                    else:
                        print(f"{Fore.RED}✗ Port must be between 1 and 65535{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}✗ Invalid port number{Style.RESET_ALL}")
        return default_port
    else:
        default_ports = config_info["ports"]
        print(f"  Default ports: {', '.join(map(str, default_ports))}")
        customize = get_input("Use custom ports? (y/n)", "n").lower()
        
        if customize == "y":
            ports = []
            for i, default_port in enumerate(default_ports):
                while True:
                    try:
                        port = int(get_input(f"Enter port {i+1}/{len(default_ports)}", str(default_port)))
                        if 1 <= port <= 65535:
                            ports.append(port)
                            break
                        else:
                            print(f"{Fore.RED}✗ Port must be between 1 and 65535{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}✗ Invalid port number{Style.RESET_ALL}")
            return ports
    return default_ports


def get_duration_override_seconds():
    """Ask user if they want to force 20 min duration"""
    print_section("Duration Override")
    choice = get_input("Force duration to 20 minutes (1200s)? (y/n)", "n").lower()
    if choice == "y":
        return 1200
    return None


def check_dependencies():
    """Check if required Python packages are installed"""
    print_section("Dependency Check")
    
    required_packages = [
        ("pydantic", "pydantic"),
        ("aiohttp", "aiohttp"),
        ("colorama", "colorama"),
        ("yaml", "PyYAML")
    ]
    optional_packages = [
        ("psutil", "psutil")
    ]
    
    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {package_name}")
        except ImportError:
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        print(f"\n{Fore.RED}✗ Missing dependencies: {', '.join(missing)}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Install them with:{Style.RESET_ALL}")
        print(f"  pip install {' '.join(missing)}")
        print(f"\n{Fore.YELLOW}Or install all requirements:{Style.RESET_ALL}")
        print(f"  pip install -r requirements.txt")
        return False
    
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {package_name} (optional)")
        except ImportError:
            print(f"  {Fore.YELLOW}⚠{Style.RESET_ALL} {package_name} (optional, missing)")

    print(f"\n{Fore.GREEN}✓ All dependencies installed{Style.RESET_ALL}")
    return True


def get_webui_ip():
    """Get Open WebUI VPS IP from user"""
    print_section("Open WebUI VPS Configuration")
    
    while True:
        webui_ip = get_input("Enter Open WebUI VPS IP address", "127.0.0.1")
        if validate_ip(webui_ip):
            return webui_ip
        else:
            print(f"{Fore.RED}✗ Invalid IP address. Please try again.{Style.RESET_ALL}")


def select_test():
    """Let user select test configuration"""
    display_test_menu()
    
    while True:
        choice = get_input("\nSelect test configuration (1-7)", "1")
        if choice in TEST_CONFIGS:
            return choice
        else:
            print(f"{Fore.RED}✗ Invalid choice. Please select 1-7.{Style.RESET_ALL}")


def create_dynamic_config(gpu_ip, test_choice, custom_ports=None, duration_override_seconds=None):
    """Create a temporary config file with user-specified IPs and ports"""
    config_info = TEST_CONFIGS[test_choice]
    
    # For custom config, just return the file path
    if test_choice == "5":
        custom_path = get_input("Enter path to custom config file")
        if not os.path.exists(custom_path):
            print(f"{Fore.RED}✗ Config file not found: {custom_path}{Style.RESET_ALL}")
            sys.exit(1)
        if not duration_override_seconds:
            return custom_path
        # Create a temp copy with overridden duration to avoid mutating user file
        try:
            with open(custom_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"{Fore.RED}✗ Error loading config: {e}{Style.RESET_ALL}")
            sys.exit(1)
        if "workload" in config:
            config["workload"]["duration_seconds"] = duration_override_seconds
        temp_dir = tempfile.gettempdir()
        temp_config_path = os.path.join(temp_dir, f"llm_stress_test_{test_choice}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return temp_config_path
    
    # Load the template config
    template_path = config_info["file"]
    try:
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}✗ Template config not found: {template_path}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Update IP addresses and ports in config
    if config_info["single_server"]:
        # Single server config
        if "server" in config:
            port = custom_ports if custom_ports else config_info["port"]
            config["server"]["base_url"] = f"http://{gpu_ip}:{port}/completion"
    else:
        # Multi-server config
        if "servers" in config:
            ports = custom_ports if custom_ports else config_info["ports"]
            for i, server in enumerate(config["servers"]):
                if i < len(ports):
                    port = ports[i]
                    server["base_url"] = f"http://{gpu_ip}:{port}/completion"
    
    # Apply duration override if requested
    if duration_override_seconds and "workload" in config:
        config["workload"]["duration_seconds"] = duration_override_seconds

    # Create temporary config file
    temp_dir = tempfile.gettempdir()
    temp_config_path = os.path.join(temp_dir, f"llm_stress_test_{test_choice}.yaml")
    
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config_path


def verify_connectivity(gpu_ip, test_choice, custom_ports=None):
    """Verify connectivity to GPU server"""
    print_section("Connectivity Check")
    
    config_info = TEST_CONFIGS[test_choice]
    
    if test_choice == "5":
        print(f"{Fore.YELLOW}⚠ Skipping connectivity check for custom config{Style.RESET_ALL}")
        return True
    
    ports_to_check = []
    if config_info["single_server"]:
        ports_to_check = [custom_ports if custom_ports else config_info["port"]]
    else:
        ports_to_check = custom_ports if custom_ports else config_info["ports"]
    
    all_ok = True
    for port in ports_to_check:
        print(f"  Checking {gpu_ip}:{port}... ", end="", flush=True)
        if check_connectivity(gpu_ip, port):
            print(f"{Fore.GREEN}✓ Reachable{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Not reachable{Style.RESET_ALL}")
            all_ok = False
    
    if not all_ok:
        print(f"\n{Fore.YELLOW}⚠ Warning: Some ports are not reachable.")
        print(f"  This might be due to firewall rules or the service not running.")
        print(f"  Continuing anyway...{Style.RESET_ALL}")
        
        proceed = get_input("\nDo you want to continue? (y/n)", "y").lower()
        if proceed != "y":
            print(f"{Fore.RED}✗ Test cancelled by user{Style.RESET_ALL}")
            sys.exit(0)
    
    return True


def display_summary(gpu_ip, webui_ip, test_choice, config_path):
    """Display test configuration summary"""
    print_section("Test Configuration Summary")
    
    config_info = TEST_CONFIGS[test_choice]
    
    print(f"  {Fore.WHITE}Test Type:{Style.RESET_ALL} {config_info['name']}")
    print(f"  {Fore.WHITE}GPU Server:{Style.RESET_ALL} {gpu_ip}")
    print(f"  {Fore.WHITE}WebUI VPS:{Style.RESET_ALL} {webui_ip}")
    print(f"  {Fore.WHITE}Config File:{Style.RESET_ALL} {config_path}")
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            duration = cfg.get("workload", {}).get("duration_seconds")
            if duration:
                print(f"  {Fore.WHITE}Duration:{Style.RESET_ALL} {duration}s")
    except Exception:
        pass
    
    print(f"\n{Fore.GREEN}✓ Configuration ready{Style.RESET_ALL}")


async def run_test(config_path, output_dir="results"):
    """Run the stress test"""
    print_section("Starting Load Test")
    
    # Import here to avoid circular dependencies
    from src.config.schema import GlobalConfig
    from src.engine.orchestrator import LoadTestOrchestrator
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load config
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        print(f"{Fore.RED}✗ Error loading config: {e}{Style.RESET_ALL}")
        return False
    
    # Validate config
    try:
        config = GlobalConfig(**raw_config)
    except Exception as e:
        print(f"{Fore.RED}✗ Configuration validation failed: {e}{Style.RESET_ALL}")
        return False
    
    def validate_config(cfg: GlobalConfig):
        if cfg.workload.duration_seconds != 1200:
            logger.warning("duration_seconds != 1200s (20 min). Requirement not enforced by code.")
        if cfg.workload.iterations < 1:
            logger.warning("iterations < 1; no consecutive runs will occur.")
        if cfg.prompts.strategy == "uniform":
            logger.warning("prompts.strategy is 'uniform'; progressive input sizing is not enabled.")
        if cfg.prompts.strategy == "staged" and not cfg.prompts.stages:
            logger.warning("prompts.strategy is 'staged' but stages are empty.")
        if cfg.prompts.strategy in ("linear", "exponential") and not cfg.prompts.ramp:
            logger.warning("prompts.strategy is ramped but prompts.ramp is not set; defaulting to min/max.")

    validate_config(config)

    # Create output directory
    run_ts = dt.now().strftime("%Y%m%d_%H%M%S")
    base_output = os.path.join(output_dir, run_ts)
    os.makedirs(base_output, exist_ok=True)
    combined_csv_path = os.path.join(base_output, "combined_results.csv")

    async def execute_run(conf: GlobalConfig, label: str, run_id: str, iteration: int):
        orchestrator = LoadTestOrchestrator(conf)
        try:
            await orchestrator.run()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠ Test interrupted by user{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}✗ Test failed: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
        finally:
            orchestrator.save_report(
                base_output,
                label=label,
                csv_path=combined_csv_path,
                include_header=not os.path.exists(combined_csv_path),
                run_id=run_id,
                iteration=iteration
            )
        return orchestrator.stats.calculate_summary()

    summaries = []
    total_iterations = config.workload.iterations
    try:
        for iter_idx in range(total_iterations):
            iter_label = f"iter{iter_idx + 1}"
            if config.comparison_mode and config.servers:
                for server in config.get_servers():
                    single_config = config.model_copy(update={
                        "server": server,
                        "servers": None,
                        "comparison_mode": False
                    })
                    run_id = f"{server.name}_{iter_label}"
                    summary = await execute_run(
                        single_config,
                        label=f"{server.name}_{iter_label}",
                        run_id=run_id,
                        iteration=iter_idx + 1
                    )
                    summaries.append({"model": server.name, "iteration": iter_idx + 1, "summary": summary})
            else:
                model_name = config.server.name if config.server else "default"
                run_id = f"{model_name}_{iter_label}"
                summary = await execute_run(
                    config,
                    label=iter_label,
                    run_id=run_id,
                    iteration=iter_idx + 1
                )
                summaries.append({"model": model_name, "iteration": iter_idx + 1, "summary": summary})
    except KeyboardInterrupt:
        return False

    if config.comparison_mode and len(set(s["model"] for s in summaries)) > 1:
        def mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else None

        per_model = {}
        for entry in summaries:
            model = entry["model"]
            summ = entry["summary"]
            per_model.setdefault(model, []).append(summ)

        rows = []
        for model, sums in per_model.items():
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "latency_p90_mean",
                "metric_value": mean([s.latency_p90 for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "ttft_p50_mean",
                "metric_value": mean([s.ttft_p50 for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "tps_mean",
                "metric_value": mean([s.global_throughput_tokens_per_sec for s in sums]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "error_rate_mean",
                "metric_value": mean([
                    (s.failed_requests / s.total_requests) if s.total_requests else None
                    for s in sums
                ]),
            })
            rows.append({
                "record_type": "comparison",
                "run_id": f"{model}_aggregate",
                "ts": dt.now().timestamp(),
                "label": "aggregate",
                "model": model,
                "metric_name": "stability_score_mean",
                "metric_value": mean([s.stability_score for s in sums]),
            })

        if rows:
            with open(combined_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LoadTestOrchestrator.CSV_COLUMNS)
                writer.writerows(rows)

    print(f"\n{Fore.GREEN}✓ Test completed successfully{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Results saved to: {combined_csv_path}{Style.RESET_ALL}")
    return True


async def main():
    """Main CLI flow"""
    print_header()
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Get GPU server IP
    gpu_ip = get_gpu_ip()
    
    # Get WebUI VPS IP
    webui_ip = get_webui_ip()
    
    # Select test configuration
    print_section("Test Selection")
    test_choice = select_test()
    
    # Get custom ports if needed
    custom_ports = get_custom_ports(test_choice)

    # Optional duration override
    duration_override_seconds = get_duration_override_seconds()
    
    # Create dynamic config
    config_path = create_dynamic_config(
        gpu_ip,
        test_choice,
        custom_ports,
        duration_override_seconds=duration_override_seconds
    )
    
    # Verify connectivity
    verify_connectivity(gpu_ip, test_choice, custom_ports)
    
    # Display summary
    display_summary(gpu_ip, webui_ip, test_choice, config_path)
    
    # Confirm before running
    print()
    proceed = get_input("Start the test? (y/n)", "y").lower()
    if proceed != "y":
        print(f"{Fore.YELLOW}✗ Test cancelled by user{Style.RESET_ALL}")
        sys.exit(0)
    
    # Run the test
    success = await run_test(config_path)
    
    # Cleanup temporary config if needed
    if config_path.startswith(tempfile.gettempdir()):
        try:
            os.remove(config_path)
        except Exception:
            pass
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}✗ Interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
