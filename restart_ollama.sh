#!/bin/bash

# Function to start ollama instances
start_ollama() {
  local gpu_id=$1
  local port=$2
  CUDA_VISIBLE_DEVICES=$gpu_id OLLAMA_HOST=127.0.0.1:$port ollama serve > ollama_$port.log 2>&1 &
  echo "Started ollama on GPU $gpu_id, port $port, PID: $!"
  # Store the PID in an array
  pids[$gpu_id]=$!
}

# Array to store PIDs
declare -A pids

# Start ollama instances
start_ollama 1 11435
start_ollama 2 11436
start_ollama 3 11437
start_ollama 4 11438
start_ollama 5 11439
start_ollama 6 11440
start_ollama 7 11441
start_ollama 8 11442
start_ollama 9 11443

# Wait for one hour (3600 seconds)
echo "Waiting for one hour..."
sleep 3600

# Kill ollama instances
echo "Killing ollama instances..."
for gpu_id in "${!pids[@]}"; do
  if kill -0 "${pids[$gpu_id]}" 2> /dev/null; then  # Check if process exists before killing
        kill "${pids[$gpu_id]}"
        echo "Killed ollama process with PID ${pids[$gpu_id]} (GPU $gpu_id)"
    else
        echo "Process with PID ${pids[$gpu_id]} (GPU $gpu_id) not found."
  fi
done

# Clear the pids array
unset pids

# Restart ollama instances (using the same function)
echo "Restarting ollama instances..."
start_ollama 1 11435
start_ollama 2 11436
start_ollama 3 11437
start_ollama 4 11438
start_ollama 5 11439
start_ollama 6 11440
start_ollama 7 11441
start_ollama 8 11442
start_ollama 9 11443

echo "Ollama instances restarted."

exit 0
