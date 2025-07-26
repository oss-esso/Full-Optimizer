# OQI VRP Environment Setup Guide

This guide provides multiple ways to set up the OQI VRP optimization environment with all required dependencies.

## Quick Setup Options

### Option 1: Using Conda (Recommended)
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate oqi_vrp
```

### Option 2: Using pip
```bash
# Create virtual environment
python -m venv oqi_env
source oqi_env/bin/activate  # On Windows: oqi_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Development Setup
```bash
# For development with additional tools
pip install -r requirements.txt
pip install -e .  # If setup.py exists
```

## Core Dependencies Overview

### Optimization & Mathematical Computing
- **OR-Tools** (>=9.5.0): Google's optimization suite for VRP solving
- **PuLP** (>=2.7.0): Linear programming optimization
- **NetworkX** (>=3.0): Graph algorithms and network analysis
- **NumPy** (>=1.24.0): Numerical computing foundation
- **SciPy** (>=1.10.0): Scientific computing algorithms

### Data Processing & Analysis
- **Pandas** (>=2.0.0): Data manipulation and analysis
- **OpenPyXL** (>=3.1.0): Excel file reading/writing
- **SQLAlchemy** (>=2.0.0): Database ORM and connectivity

### Visualization & Mapping
- **Matplotlib** (>=3.7.0): Basic plotting
- **Seaborn** (>=0.12.0): Statistical visualization
- **Plotly** (>=5.15.0): Interactive plots
- **Folium** (>=0.14.0): Interactive maps
- **GeoPandas** (>=0.13.0): Geographic data analysis

### Form Generation & Documents
- **fpdf2** (>=2.8.0): PDF generation
- **QRCode** (>=8.2): QR code generation for forms
- **ReportLab** (>=4.0.0): Advanced PDF reporting

### Web Services & APIs
- **Requests** (>=2.31.0): HTTP library for API calls
- **FastAPI** (>=0.103.0): Modern web framework
- **Uvicorn** (>=0.23.0): ASGI server

### Development & Testing
- **Pytest** (>=7.4.0): Testing framework
- **Black** (>=23.7.0): Code formatting
- **MyPy** (>=1.5.0): Static type checking

## Environment Verification

After installation, verify your setup:

```python
# Test core VRP dependencies
import numpy as np
import pandas as pd
import ortools
import pulp
import networkx as nx
import folium
import matplotlib.pyplot as plt

# Test form generation
import fpdf
import qrcode

# Test optimization
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

print("✅ All core dependencies installed successfully!")
```

## Troubleshooting

### Common Issues

1. **OR-Tools Installation Issues**
   ```bash
   # Try specific version
   pip install ortools==9.5.2237
   ```

2. **GeoPandas Dependencies**
   ```bash
   # Install system dependencies first
   conda install geos proj gdal
   pip install geopandas
   ```

3. **Excel Reading Issues**
   ```bash
   # Ensure both openpyxl and xlrd are installed
   pip install openpyxl xlrd
   ```

4. **PDF Generation Issues**
   ```bash
   # Use fpdf2 instead of fpdf
   pip install fpdf2 --upgrade
   ```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8 GB (16 GB recommended)
- **CPU**: 4 cores (8 cores recommended)
- **Storage**: 5 GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended for Large Instances
- **RAM**: 32 GB or more
- **CPU**: 16 cores or more
- **Storage**: SSD with 20 GB free space

## Performance Optimization

### For Large VRP Instances
```bash
# Install performance packages
pip install memory-profiler py-spy
pip install pymoo  # Additional optimization algorithms
```

### Parallel Processing
```bash
# Ensure multiprocessing support
pip install joblib psutil
```

## Docker Alternative

For containerized deployment:

```dockerfile
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["python", "src/main.py"]
```

## Version Compatibility

- **Python**: 3.11+ (recommended), 3.9+ (minimum)
- **OR-Tools**: 9.5+ for latest VRP features
- **Pandas**: 2.0+ for enhanced performance
- **NumPy**: 1.24+ for latest mathematical functions

## License Considerations

Most dependencies are open-source with permissive licenses:
- MIT: NumPy, Pandas, Requests, FastAPI
- Apache 2.0: OR-Tools, TensorFlow
- BSD: SciPy, Matplotlib, Seaborn

Check individual package licenses for commercial use.

---

For support, consult the project documentation or raise an issue in the repository.
