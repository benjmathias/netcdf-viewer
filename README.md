# NetCDF Viewer

A Streamlit application for viewing and analyzing NetCDF files.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Using Poetry](#using-poetry)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Docker Deployment](#docker-deployment)
- [Development](#development)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Linting and Type Checking](#linting-and-type-checking)
  - [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

NetCDF Reader is a user-friendly Streamlit application designed to help scientists, researchers, and data analysts view and analyze NetCDF (Network Common Data Form) files. It provides an interactive interface for exploring data variables, metadata, and visualizing data through various plots.

## Features

- **File Upload:** Easily upload NetCDF files for analysis.
- **Metadata Display:** View comprehensive metadata including file size, dimensions, and global attributes.
- **Variable Exploration:** Browse and explore variables within the NetCDF file.
- **Data Visualization:** Generate interactive scatter plots and data previews using Plotly and Pandas.
- **Error Handling:** Informative error messages for a seamless user experience.
- **Scalable Performance:** Handles large datasets with automatic downsampling.

## Installation

### Prerequisites

- **Python:** 3.12+
- **Poetry:** [Poetry](https://python-poetry.org/) package manager
- **Docker:** (Optional) For containerized deployment

### Clone the Repository

```bash
git clone git@github.com:benjmathias/netcdf-reader.git
cd netcdf-reader
```

### Using Poetry

1. **Install Dependencies:**

   ```bash
   poetry install
   ```

2. **Activate the Virtual Environment:**

   ```bash
   poetry shell
   ```

## Usage

### Running the Application

You can run the Streamlit application using the provided script:

```bash
./run.sh
```

Alternatively, use Poetry to run the application:

```bash
poetry run streamlit run ./src/viewers/streamlit_viewer.py --server.headless true
```

### Docker Deployment (in progress)

1. **Build the Docker Image:**

   ```bash
   docker build -t netcdf-reader .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8501:8501 netcdf-reader
   ```

   Access the application at [http://localhost:8501](http://localhost:8501).

## Development

### Setting Up the Development Environment

1. **Install Development Dependencies:**

   ```bash
   poetry install --with dev
   ```

2. **Pre-commit Hooks:** (in progress)

   Ensure code quality by setting up pre-commit hooks.

   ```bash
   pre-commit install
   ```

### Linting and Type Checking

- **Lint with Ruff:**

  ```bash
  poetry run ruff check .
  ```

- **Type Checking with MyPy:**

  ```bash
  poetry run mypy .
  ```

### Testing (in progress)

Run the test suite using Pytest:

```bash
poetry run pytest
```

## TODO
- full typing with mypy check
- ruff
- pre-commit
- make lint
- make run
- make docker run
- make test
- sonarqube
- slider for selecting file part
- safety