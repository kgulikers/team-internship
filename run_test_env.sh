#!/usr/bin/env bash

# This script fully restarts the Conda environment and runs the test script.

# 1. Deactivate any currently active Conda environment
if [[ -n "$CONDA_PREFIX" ]]; then
    conda deactivate
fi

# 2. Ensure Conda commands are available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "WARNING: conda.sh not found; hoping conda is already on PATH."
fi

# 3. Activate the desired environment
conda activate env_isaaclab
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate 'env_isaaclab'. Make sure the environment exists."
    exit 1
fi

echo "Activated Conda environment: $(conda info --envs | grep '\* env_isaaclab')"

# 4. Change to project root
cd "$HOME/Avular/Avular" || {
    echo "ERROR: Could not cd to project directory '~/Avular/Avular'"
    exit 1
}

echo "Running test_env.py in $(pwd)"

# 5. Execute the test script, replacing this process
exec python3 scripts/test_env.py
