#!/bin/bash

# Configuration
REMOTE_HOST="24.124.32.70"
REMOTE_PORT="8080"
CONFIG_FILE="config/remote_gpu.yaml"

echo "=== LLM Stress Test: Remote GPU (Direct Connect) ==="
echo "Target: http://$REMOTE_HOST:$REMOTE_PORT"

# Check for config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# 1. Connectivity Check
echo ">> Checking connectivity to $REMOTE_HOST:$REMOTE_PORT..."

if nc -z -w 2 $REMOTE_HOST $REMOTE_PORT 2>/dev/null; then
    echo ">> Success: Port $REMOTE_PORT is reachable."
else
    echo ">> WARNING: Could not connect to $REMOTE_HOST:$REMOTE_PORT."
    echo "   Please make sure the firewall on the GPU server allows incoming traffic on port $REMOTE_PORT."
    echo "   Continuing anyway (in case ICMP is blocked but HTTP works)..."
fi

# 2. Run the Python Load Tester
echo ">> Starting Load Test..."
python main.py --config $CONFIG_FILE
