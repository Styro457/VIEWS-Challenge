# VIEWS-Challenge

-----

## Table of Contents

- [Installation](#installation)

## Project Overview
API Service that provides user friendly quering functionality for accessing VIEWS conflict forecasting system data based on simple API keys access logic

## Features
- Query available months and grid cells
- Get forecast values by: Country, Grid IDs, Range of Months etc
- Select metrics that are calculated using statistical analysis e.g. MAP, confidence intervals, etc
- Paginated JSON Response
- API Key access system

## Tech Stack
- Python
- FastAPI
- Pytest
- Ruff
- Makefile

## Endpoints
FastAPI provides dynamicly updated **/docs** endpoint that helps with basic documentation and testing of endpoints

### Data Endpoints
- **/months** - returns an id list of all available months
- **/countries** - returns an id list for all available countries
- **/all_cells** - returns a id list of all available grid cells
- **/cells** - accepts cells identifiers from the user (e.g. cell id, country id), filters the dataset based on them, returns calculated values requested by the user (e.g.  MAP (mean value), 3 confidence intervals (50%, 90%, 99%), and 6 probabilities of passing certain thresholds (plus a few helper values like grid ID, location, country ID))

### API Keys Endpoints
- **/create_user_api_key** - Generates a standart api key and saves it to the database
- **/create_admin_api_key** - Generates an admin api key and saves it to the database, requires admin key to be accessed
- **/get_key_info/{key}** - Fetches information about the provided api key from the database
- **/revoke_key/{key}** - Revokes provided api key, requires admin key to be accessed



## Setup Instructions

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

## Admin Access Key
SviyiqdT_sANAmbD0qTs_NJXH4xXYWBgb0twevfMVYE

### Example
curl -H "Authorization: SviyiqdT_sANAmbD0qTs_NJXH4xXYWBgb0twevfMVYE" http://localhost:8000/admin_stuff

### How to create an admin key
```
sudo -u postgres psql views_api

INSERT INTO api_keys (key, expires_at, revoked, daily_limit, admin, expired)
VALUES (
  'SviyiqdT_sANAmbD0qTs_NJXH4xXYWBgb0twevfMVYE',
  NULL,
  FALSE,
  1000,  -- or whatever DEFAULT_DAILY_REQUESTS_LIMIT is
  TRUE,
  FALSE
);
```

