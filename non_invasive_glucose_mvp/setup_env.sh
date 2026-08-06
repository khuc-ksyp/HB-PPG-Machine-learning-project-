#!/usr/bin/env bash
# Virtual Environment Setup Script for Non-Invasive Blood Glucose Estimator
set -e

echo "=== Initializing Python Virtual Environment ==="
python -m venv venv

echo "=== Activating Virtual Environment ==="
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== Environment Setup Complete ==="
