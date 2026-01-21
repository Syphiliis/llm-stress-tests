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
        choice = get_input("\nSelect test configuration (1-5)", "1")
        if choice in TEST_CONFIGS:
            return choice
        else:
            print(f"{Fore.RED}✗ Invalid choice. Please select 1-5.{Style.RESET_ALL}")


def create_dynamic_config(gpu_ip, test_choice):
    """Create a temporary config file with user-specified IPs"""
    config_info = TEST_CONFIGS[test_choice]
    
    # For custom config, just return the file path
    if test_choice == "5":
        custom_path = get_input("Enter path to custom config file")
        if not os.path.exists(custom_path):
            print(f"{Fore.RED}✗ Config file not found: {custom_path}{Style.RESET_ALL}")
            sys.exit(1)
        return custom_path
    
    # Load the template config
    template_path = config_info["file"]
    try:
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}✗ Template config not found: {template_path}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Update IP addresses in config
    if config_info["single_server"]:
        # Single server config
        if "server" in config:
            port = config_info["port"]
            config["server"]["base_url"] = f"http://{gpu_ip}:{port}/completion"
    else:
        # Multi-server config
        if "servers" in config:
            for i, server in enumerate(config["servers"]):
                if i < len(config_info["ports"]):
                    port = config_info["ports"][i]
                    server["base_url"] = f"http://{gpu_ip}:{port}/completion"
    
    # Create temporary config file
    temp_dir = tempfile.gettempdir()
    temp_config_path = os.path.join(temp_dir, f"llm_stress_test_{test_choice}.yaml")
    
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config_path


def verify_connectivity(gpu_ip, test_choice):
    """Verify connectivity to GPU server"""
    print_section("Connectivity Check")
    
    config_info = TEST_CONFIGS[test_choice]
    
    if test_choice == "5":
        print(f"{Fore.YELLOW}⚠ Skipping connectivity check for custom config{Style.RESET_ALL}")
        return True
    
    ports_to_check = []
    if config_info["single_server"]:
        ports_to_check = [config_info["port"]]
    else:
        ports_to_check = config_info["ports"]
    
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and run orchestrator
    orchestrator = LoadTestOrchestrator(config)
    
    try:
        await orchestrator.run()
        orchestrator.save_report(output_dir)
        print(f"\n{Fore.GREEN}✓ Test completed successfully{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Results saved to: {output_dir}{Style.RESET_ALL}")
        return True
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠ Test interrupted by user{Style.RESET_ALL}")
        orchestrator.save_report(output_dir)
        return False
    except Exception as e:
        print(f"\n{Fore.RED}✗ Test failed: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main CLI flow"""
    print_header()
    
    # Get GPU server IP
    gpu_ip = get_gpu_ip()
    
    # Get WebUI VPS IP
    webui_ip = get_webui_ip()
    
    # Select test configuration
    print_section("Test Selection")
    test_choice = select_test()
    
    # Create dynamic config
    config_path = create_dynamic_config(gpu_ip, test_choice)
    
    # Verify connectivity
    verify_connectivity(gpu_ip, test_choice)
    
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
