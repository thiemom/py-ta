# py-ta

A collection of Python scripts for thermo-acoustics analysis and modeling.

## Overview

This repository contains standalone Python scripts useful for thermo-acoustics research and engineering. The scripts are designed to be used as-is for specific analysis tasks rather than as part of an integrated tool suite.

## Contents

- **`pyftf.py`** - DTL (Distributed Time Lag) Flame Transfer Function model implementation with parameter fitting capabilities
- **`example_usage.py`** - Example demonstrating FTF model usage with synthetic data

## Usage

1. Set up the Python environment:
   ```bash
   ./setup_environment.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the scripts directly:
   ```bash
   python py-ftf.py
   python example_usage.py
   ```

## Dependencies

- numpy
- pandas  
- matplotlib
- scipy

All dependencies are listed in `requirements.txt` and installed automatically by the setup script.

## License

See `LICENSE` file for details.