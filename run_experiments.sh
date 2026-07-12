#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Seeds to be used
SEEDS=(100 200 300)

# Scripts to execute
SCRIPTS=("src/IPPO.py" "src/MAPPO.py")

# Number of iterations for convergence demonstration
ITERATIONS=40

# Get the absolute path to the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TODAY=$(date +%Y-%m-%d)



for SCRIPT in "${SCRIPTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        # Determine algorithm name from script name (e.g. "IPPO" from "src/IPPO.py")
        ALGO=$(basename "$SCRIPT" .py)
        
        # Expected checkpoint directory based on our predictable naming convention
        EXP_DIR="$DIR/A-checkpoints/$TODAY/${ALGO}_seed_${SEED}"
        
        echo "------------------------------------------------------"
        
        if [ -d "$EXP_DIR" ]; then
            echo "Found existing directory $EXP_DIR."
            echo "Attempting to RESUME: python $SCRIPT --seed $SEED --iterations $ITERATIONS --resume $EXP_DIR"
            python "$DIR/$SCRIPT" --seed "$SEED" --iterations "$ITERATIONS" --resume "$EXP_DIR"
        else
            echo "Starting NEW run: python $SCRIPT --seed $SEED --iterations $ITERATIONS"
            python "$DIR/$SCRIPT" --seed "$SEED" --iterations "$ITERATIONS"
        fi
        
        echo "Completed: python $SCRIPT --seed $SEED"
        echo ""
    done
done

