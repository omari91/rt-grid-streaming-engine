"""
Entry point for the RT Grid Streaming Engine.

This is a thin wrapper around engine.main() — kept as a separate file only
because the README documents `python main.py` as the run command.

All logic (data ingestion, physics, controllers, benchmarking, figure
generation) lives in engine.py so there is a single source of truth.

Usage
-----
    # Reads data/redispatch_1yr.csv if present; falls back to a synthetic
    # Beta-distributed stream otherwise (results will not match the paper):
    python main.py
"""
from engine import main

if __name__ == "__main__":
    main()
