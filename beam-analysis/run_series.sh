#!/bin/bash

# --- Configuration ---
START_LAYOUT=0
END_LAYOUT=39
CONFIG_FILE="configs/base_config.yml"

# --- Environment Setup ---
echo "=========================================================="
echo "Starting Sequential SPECT Analysis Pipeline"
echo "Start Time: $(date)"
echo "Config Path: $CONFIG_FILE"
echo "Processing Layouts: $START_LAYOUT to $END_LAYOUT"
echo "=========================================================="

# Activate your Python virtual environment
source /vscratch/grp-rutaoyao/sid/venv/bin/activate

# --- Main Processing Loop ---
for LAYOUT_IDX in $(seq $START_LAYOUT $END_LAYOUT); do
    echo ""
    echo "----------------------------------------------------------"
    echo "Processing Layout Index: $LAYOUT_IDX"
    echo "----------------------------------------------------------"

    # --- STEP 1: Mask Extraction ---
    echo "Step 1/3: Extracting Beam Masks..."
    python arg_extract_beam_masks.py $LAYOUT_IDX --config $CONFIG_FILE
    if [ $? -ne 0 ]; then 
        echo "Error: Mask extraction failed for layout $LAYOUT_IDX"
        exit 1
    fi

    # --- STEP 2: Property Extraction ---
    echo "Step 2/3: Extracting Beam Properties (FWHM, Sensitivity)..."
    python arg_extract_beam_properties.py $LAYOUT_IDX --config $CONFIG_FILE
    if [ $? -ne 0 ]; then 
        echo "Error: Property extraction failed for layout $LAYOUT_IDX"
        exit 1
    fi

    # --- STEP 3: ASCI Histogram Generation ---
    echo "Step 3/3: Generating Angular Sensitivity Histogram..."
    python arg_analyze_extracted_properties.py $LAYOUT_IDX --config $CONFIG_FILE
    if [ $? -ne 0 ]; then 
        echo "Error: ASCI generation failed for layout $LAYOUT_IDX"
        exit 1
    fi

done

echo ""
echo "=========================================================="
echo "End Time:   $(date)"
echo "Pipeline Finished Successfully for all layouts!"
echo "=========================================================="