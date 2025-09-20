# VIEWS-Challenge

-----

## Table of Contents

- [Installation](#installation)

## Installation

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

