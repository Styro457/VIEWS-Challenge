# VIEWS Simple API
-----

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Endpoints](#endpoints)

# Overview
**VIEWS Simple API** is a REST API that offers **public access** to the data from the **VIEWS conflict forecasting system**.

Using the raw prediction data from VIEWS and the [views_pipeline_core](https://github.com/views-platform/views-pipeline-core/tree/main) module for statystical analasys it simplifies the process of handling the data and directly gives the user the information they need.

## ⚙️ Features
- **Searchable** by Country, Grid IDs and Range of Months
- **Precise selection of metrics** returning only relevant information
- **Optimization** for large amounts of data using **compression**, **caching** and **pagination**
- Simple **Access-Key** system for **access rights** and **rate-limiting**
- **Fully Conteinerized** and **easy deployment**
- **Highly customizable**

## Planned Features
- Database for storing pre-computed data from multiple sources
- Endpoints for returning interactive graphs
- More statistics

## ⚡ Development

**VIEWS Simple API** is built with a modern Python stack, designed for performance, maintainability, and easy deployment. The codebase emphasizes clean, testable, and efficient practices, making it simple to extend and customize for different use cases.  

### Tech Stack
- **Python** – Core language for API logic and data processing  
- **FastAPI** – High-performance REST API framework  
- **Pytest** – Framework for automated testing  
- **Ruff** – Linter for consistent code quality  
- **Makefile** – Automates common development tasks  
- **Docker** – Containerization for consistent deployment

# Installation

This porject required Python3.11 to run.
Follow these steps to get it running on your system.  

## 🛠️ Quick Install

1. **Set up a Python 3.11 virtual environment**
  ```bash
   python3.11 -m venv venv

   source venv/bin/activate  # Linux/Mac

   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
```bash
make install
   ```

4. **Import data**

     Add `preds_001.parquet` in `/env`

6. **Run the API locally**  
  ```bash
   make run
   ```

## 🐳 Docker Installation

1. **Build the Docker image**  
   ```bash
   sudo docker build -t views_challenge:latest .
   ```

2. **Run the Docker container**  
   ```bash
   sudo docker run -p 8000:8000 views_challenge:latest
   ```

## 🧰 Development Tasks

- **Check code linting** using ruff
  ```bash
  make lint
  ```

- **Run automated tests** using pytest
  ```bash
  make test
  ```

# Configuration 

The API requires environment variables to run properly. A sample configuration file is provided in the repository.  

1. **Copy the example file**  
   ```bash
   cp .env.example .env
   ```

2. **Update the `.env` file** with your own settings

The application will automatically load variables from the `.env` file when starting up.

❗ **In order to use api-keys you need to have `key_mode` set to `true` and have valid database credentials**


# Endpoints
FastAPI provides dynamicly updated **/docs** endpoint that helps with basic documentation and testing of endpoints

### Data Endpoints
- **/months** - returns an id list of all available months
- **/countries** - returns an id list for all available countries
- **/all_cells** - returns a id list of all available grid cells
- **/cells** - accepts cells identifiers from the user (e.g. cell id, country id), filters the dataset based on them, returns calculated values requested by the user (e.g.  MAP (mean value), 3 confidence intervals (50%, 90%, 99%), and 6 probabilities of passing certain thresholds (plus a few helper values like grid ID, location, country ID))

### Example Queries for /Cells
- **GET /cells?return_params=map_value,ci_90,ci_99** - return MAP estimates and 90/99% confidence intervals for all cells (limit may apply)
- **GET /cells?country_id=45&violence_types=sb,ns&return_params=map_value,prob_above_050** - return MAP and probability of violence exceeding 0.5 for cells with country id equal to 49

### API Keys Endpoints
- **/create_user_api_key** - Generates a standart api key and saves it to the database
- **/create_admin_api_key** - Generates an admin api key and saves it to the database, requires admin key to be accessed
- **/get_key_info/{key}** - Fetches information about the provided api key from the database
- **/revoke_key/{key}** - Revokes provided api key, requires admin key to be accessed
