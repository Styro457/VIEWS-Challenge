# VIEWS Simple API
![GitHub Release](https://img.shields.io/github/v/release/orangasus/VIEWS-Challenge)
![GitHub License](https://img.shields.io/github/license/orangasus/VIEWS-Challenge)
-----

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)

# Overview
**VIEWS Simple API** is a REST API that offers **public access** to the data from the **VIEWS conflict forecasting system**.

Using the raw prediction data from VIEWS and the [views_pipeline_core](https://github.com/views-platform/views-pipeline-core/tree/main) module it simplifies the process of handeling the data and directly gives the user the information they need.

## ⚙️ Features
- **Searchable** by Country, Grid IDs and Range of Months
- **Precise selection of metrics** returning only relevant information
- **Optimization** for large amounts of data using **compression**, **caching** and **pagination**
- Simple **Access-Key** system for **access rights** and **rate-limiting**
- **Fully Conteinerized** and **easy deployment**
- **Highly customizable**

## Tech Stack
- Python
- FastAPI
- Pytest
- Ruff
- Makefile
- Docker


# Installation

```console
pip install -e .[dev]
```

## Running

```console
uvicorn views_challenge.main:app --reload
```

## Tasks

Lint
```console
ruff check src/views_challenge tests
```

## Test Dataset


## Endpoints
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
