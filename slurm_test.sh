#! /bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

# 629 for target 48kHz
CHECKPOINT=629
# Array of sample rates
SAMPLE_RATES=(8000 12000 16000 24000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python for_test.py -r $CHECKPOINT --sr $SR --save --cuda
done

# 584 for target 16kHz
CHECKPOINT=584
# Array of sample rates
SAMPLE_RATES=(2000 4000 8000 12000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python for_test.py -r $CHECKPOINT --sr $SR --save --cuda
done