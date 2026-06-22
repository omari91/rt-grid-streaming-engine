# rt-grid-streaming-engine

Reference implementation for the paper: *"Towards the Digital Twin: A Streaming AC Power Flow Pipeline for Real-Time Grid Control."*

This repository provides a vectorized Forward-Backward Sweep (FBS) solver and control kernel for sub-cycle (0.03ms) distribution grid simulation and Redispatch management.

## Key Features
- **Deterministic Latency:** High-performance vectorized physics kernel.
- **Physics-Aware Control:** Resolves non-linear AC voltage stability constraints.
- **Live Ingestion:** Real-time integration via Netztransparenz API.
- **Reproducible:** Calibration via BDEW/BNetzA statistical parameters.

## Methodology
The core physics engine resolves non-linear AC voltage stability constraints by treating the 50-bus radial MV feeder as an aggregate. It utilizes a highly optimized Forward-Backward Sweep (FBS) solver that bypasses the heavy matrix inversions required by traditional Newton-Raphson. This enables sub-cycle (0.03ms) determinism. The pipeline incorporates a `VoltageAwareController` for Redispatch management and handles real-time live ingestion via the Netztransparenz API.

## Prerequisites
- Python 3.8+
- Operating System: Windows, macOS, or Linux
- API Credentials for Netztransparenz (Optional for local testing, synthetic fallback included)

## Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/omari91/rt-grid-streaming-engine.git
   cd rt-grid-streaming-engine
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables (if testing live API):** 
   - `export NTP_CLIENT_ID="your_id"`
   - `export NTP_CLIENT_SECRET="your_secret"`
4. **Run the simulation benchmark:**
   ```bash
   python main.py
   ```
   *Note: All output figures and performance logs will be automatically saved to the `final_output/` directory.*

## Citation
If you use this code in your research, please consider citing our work. The full citation for the corresponding IEEE paper will be added upon publication.
