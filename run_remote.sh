#!/bin/bash

# Configuration
REMOTE_USER="root"
REMOTE_HOST="24.124.32.70"
REMOTE_SSH_PORT="38700"
REMOTE_MODEL_PORT="8080" # Port where the model is listening on the remote server
LOCAL_PORT="8080"
CONFIG_FILE="config/remote_gpu.yaml"

echo "=== LLM Stress Test: Remote GPU Setup ==="
echo "Target: $REMOTE_USER@$REMOTE_HOST:$REMOTE_SSH_PORT"
echo "Tunnel: Localhost:$LOCAL_PORT -> Remote:$REMOTE_MODEL_PORT"

# Check for config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found."
    exit 1
fi

# 1. Establish SSH Tunnel
echo ">> establishing SSH tunnel..."
# -N: Do not execute a remote command.
# -L: Specifies that connections to the given TCP port or Unix socket on the local (client) host are to be forwarded to the given host and port, or Unix socket, on the remote side.
# -f: Requests ssh to go to background just before command execution.
ssh -f -N -L $LOCAL_PORT:localhost:$REMOTE_MODEL_PORT -p $REMOTE_SSH_PORT $REMOTE_USER@$REMOTE_HOST

if [ $? -ne 0 ]; then
    echo "Error: Failed to establish SSH tunnel."
    echo "Please check your SSH keys and connectivity."
    exit 1
fi

TUNNEL_PID=$(pgrep -f "ssh -f -N -L $LOCAL_PORT")
echo ">> Tunnel established (PID likely $TUNNEL_PID)"

# Function to cleanup tunnel on exit
cleanup() {
    echo ""
    echo ">> Cleaning up..."
    if [ ! -z "$TUNNEL_PID" ]; then
        echo ">> Killing SSH tunnel..."
        pkill -f "ssh -f -N -L $LOCAL_PORT"
    fi
    echo ">> Done."
}
trap cleanup EXIT

# 2. Wait a moment for tunnel to settle
sleep 2

# 3. Run the Python Load Tester
echo ">> Starting Load Test..."
python main.py --config $CONFIG_FILE

# Cleanup happens automatically via trap
