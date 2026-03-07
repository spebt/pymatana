#!/bin/bash

#SBATCH --job-name=spect_analysis_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=04:00:00                 # Adjusted for full pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10               # Multithreading for PPDF and arc sampling
#SBATCH --mem=2G                       # Significant RAM for HDF5 processing
#SBATCH --array=0-39                    # Adjust based on your total layout count
#SBATCH --mail-user=smehta28@buffalo.edu
#SBATCH --mail-type=FAIL,END

# --- Logging Setup ---
mkdir -p slurm_logs/out slurm_logs/err
#SBATCH --output=slurm_logs/out/analysis_%A_%a.out
#SBATCH --error=slurm_logs/err/analysis_%A_%a.err

# --- Environment Setup ---
echo "=========================================================="
echo "Start Time: $(date)"
echo "Array Task ID (Layout Index): $SLURM_ARRAY_TASK_ID"
echo "Config Path: configs/base_config.yml"
echo "=========================================================="

source /vscratch/grp-rutaoyao/sid/venv/bin/activate

# --- STEP 1: Mask Extraction ---
echo "Step 1/3: Extracting Beam Masks..."
python arg_extract_beam_masks.py $SLURM_ARRAY_TASK_ID --config configs/base_config.yml 
if [ $? -ne 0 ]; then echo "Mask extraction failed"; exit 1; fi

# --- STEP 2: Property Extraction ---
echo "Step 2/3: Extracting Beam Properties (FWHM, Sensitivity)..."
python arg_extract_beam_properties.py $SLURM_ARRAY_TASK_ID --config configs/base_config.yml 
if [ $? -ne 0 ]; then echo "Property extraction failed"; exit 1; fi

# --- STEP 3: ASCI Histogram Generation ---
echo "Step 3/3: Generating Angular Sensitivity Histogram..."
python arg_analyze_extracted_properties.py $SLURM_ARRAY_TASK_ID --config configs/base_config.yml 
if [ $? -ne 0 ]; then echo "ASCI generation failed"; exit 1; fi


echo "=========================================================="
echo "End Time:   $(date)"
echo "Pipeline Finished Successfully"
echo "=========================================================="