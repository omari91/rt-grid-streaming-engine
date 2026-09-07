#!/bin/bash
mkdir -p final_output/logs
source .venv/bin/activate
for script in regression.py run_camera_ready_analysis.py run_e2_recall_audit.py run_experiment_matrix.py run_full_year_seasonal_check.py run_latency_standalone.py run_seasonal_check.py run_severity_recall_audit.py run_severity_sweep.py run_threshold_sensitivity.py; do
    echo "Running $script..."
    python "$script" > "final_output/logs/${script}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error in $script! Check final_output/logs/${script}.log"
    else
        echo "$script finished successfully."
    fi
done
