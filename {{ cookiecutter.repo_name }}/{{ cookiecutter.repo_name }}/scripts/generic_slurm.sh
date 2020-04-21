#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user={EMAIL} # mail to me!
#SBATCH -n 1           # 1 core
#SBATCH -t 1-00:00:00   # 1 days 
#SBATCH -p {Partition Name} # Partition with GPUs
#SBATCH --mem=20000  # 20 gb 
#SBATCH -J slurm_generic # sensible name for the job
#SBATCH --output={directory_path/}slurm_generic_%j.log   # Standard output and error log

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" src/scripts/slurm_scripts/generic_slurm.sh



# Load some modules
# module load c3ddb/glibc/2.14
# module load cuda80/toolkit/8.0.44

# Activate conda
# source {path}/miniconda3/etc/profile.d/conda.sh

# Activate right python version
# conda activate {conda_env}

# Evaluate the passed in command... in this case, it should be python ... 
eval $CMD

