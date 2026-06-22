# rt-grid-streaming-engine

Reference implementation for the paper: *"Towards the Digital Twin: A Streaming AC Power Flow Pipeline for Real-Time Grid Control."*

This repository provides a vectorized Forward-Backward Sweep (FBS) solver and control kernel for sub-cycle (0.03ms) distribution grid simulation and Redispatch management.

## Key Features
- **Deterministic Latency:** High-performance vectorized physics kernel.
- **Physics-Aware Control:** Resolves non-linear AC voltage stability constraints.
- **Live Ingestion:** Real-time integration via Netztransparenz API.
- **Reproducible:** Calibration via BDEW/BNetzA statistical parameters.

## Setup
1. Set environment variables: 
   - `export NTP_CLIENT_ID="your_id"`
   - `export NTP_CLIENT_SECRET="your_secret"`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

## Citation
If you use this code in your research, please consider citing our work. The full citation for the corresponding IEEE paper will be added upon publication.
