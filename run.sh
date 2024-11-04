#!/bin/bash

# Function to check if streamlit is installed and accessible
check_streamlit() {
    if ! poetry run which streamlit >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Check if dependencies need to be installed
if [ ! -f "poetry.lock" ] || [ ! -d ".venv" ] || ! check_streamlit; then
    echo "Installing dependencies..."
    poetry lock # This is needed in some edge cases
    poetry install
fi

# Run the application
poetry run streamlit run ./src/run_netcdf_viewer.py --server.headless true