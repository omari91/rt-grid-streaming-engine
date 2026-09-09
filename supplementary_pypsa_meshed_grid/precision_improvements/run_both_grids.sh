#!/bin/bash
set -e

# Run for CIGRE
echo "Running suite for CIGRE MV grid..."
export GRID_TYPE=cigre
bash supplementary_pypsa_meshed_grid/precision_improvements/run_all_scripts.sh
mv supplementary_pypsa_meshed_grid/precision_improvements/final_output supplementary_pypsa_meshed_grid/precision_improvements/final_output_cigre

# Run for PyPSA
echo "Running suite for PyPSA-DE meshed grid..."
export GRID_TYPE=pypsa
bash supplementary_pypsa_meshed_grid/precision_improvements/run_all_scripts.sh
mv supplementary_pypsa_meshed_grid/precision_improvements/final_output supplementary_pypsa_meshed_grid/precision_improvements/final_output_pypsa

echo "Done! Both grid tests are available in final_output_cigre and final_output_pypsa."
