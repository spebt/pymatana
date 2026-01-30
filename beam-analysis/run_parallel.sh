#!/bin/bash

#SBATCH --job-name=spect_analysis_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=04:00:00                 # Adjusted for full pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # Multithreading for PPDF and arc sampling
#SBATCH --mem=32G                       # Significant RAM for HDF5 processing
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
echo "Config Path: configs/analysis_config.yml"
echo "=========================================================="

source ../venv/bin/activate

# --- STEP 1: Mask Extraction ---
echo "Step 1/5: Extracting Beam Masks..."
python arg_extract_beam_masks.py $SLURM_ARRAY_TASK_ID --config configs/analysis_config.yml 
if [ $? -ne 0 ]; then echo "Mask extraction failed"; exit 1; fi

# --- STEP 2: Property Extraction ---
echo "Step 2/5: Extracting Beam Properties (FWHM, Sensitivity)..."
python arg_extract_beam_properties.py $SLURM_ARRAY_TASK_ID --config configs/analysis_config.yml 
if [ $? -ne 0 ]; then echo "Property extraction failed"; exit 1; fi

# --- STEP 3: ASCI Histogram Generation ---
echo "Step 3/5: Generating Angular Sensitivity Histogram..."
python arg_analyze_extracted_properties.py $SLURM_ARRAY_TASK_ID --config configs/analysis_config.yml 
if [ $? -ne 0 ]; then echo "ASCI generation failed"; exit 1; fi

# --- STEP 4: Plotting (Run only on the first task to avoid redundancy) ---
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Step 4/5: Generating ASCI Map Plot..."
    # Note: Pass layout range (e.g., 0-39) if aggregating all for the final map
    python plot_asci_map.py --config configs/analysis_config.yml --layouts $(seq -s ' ' 0 39) 

    echo "Step 5/5: Generating Beam Statistics..."
    
    # Mode A: Base Layout Analysis (using layout 0)
    echo "  - Running stats for Base Layout (Position 00)..."
    python analyze_beam_statistics.py \
        --props ../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs/beams_properties_configuration_00.hdf5 \
        --masks ../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs/beams_masks_configuration_00.hdf5 \
        --config configs/analysis_config.yml \
        --out ../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs/plots/base_stats 

    # Mode B: Cumulative Analysis (Aggregating the entire folder)
    echo "  - Running cumulative stats for all layouts..."
    python analyze_beam_statistics.py \
        --props-dir ../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs/ \
        --config configs/analysis_config.yml \
        --out ../../../data/mph_hourglass_single_position_base_3mm_18pinholes_stationary/outputs/plots/cumulative_stats 
fi

echo "=========================================================="
echo "End Time:   $(date)"
echo "Pipeline Finished Successfully"
echo "=========================================================="