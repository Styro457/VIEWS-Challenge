"""
API key handling functions
"""


# cryptography module for token generation
import secrets
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from views_challenge.configs.config import settings
from views_challenge.database.models import APIKey, RequestLog
from views_challenge.configs.config import settings

from views_challenge.database.database import database

keys_router = APIRouter()


# Configuration is now handled via settings object from config.py

def use_keys() -> bool:
    return settings.keys_mode and database.get_db() is not None

def check_rate_limit(api_key: str, endpoint: str, db: Session) -> bool:
    """
    Check if the API key has exceeded the rate limit.
    Returns True if within limit, False if exceeded.
    """
    # Calculate the time window (now - RATE_LIMIT_WINDOW minutes)
    window_start = datetime.utcnow() - timedelta(minutes=settings.rate_limit_window)

    # Count requests in the current window
    request_count = (
        db.query(RequestLog)
        .filter(
            RequestLog.api_key == api_key,
            RequestLog.endpoint == endpoint,
            RequestLog.timestamp >= window_start,
        )
        .count()
    )

    # Check if within limit
    return request_count < settings.rate_limit_requests


def log_request(api_key: str, endpoint: str, db: Session):
    """Log the current request for rate limiting."""
    request_log = RequestLog(
        api_key=api_key, endpoint=endpoint, timestamp=datetime.utcnow()
    )
    db.add(request_log)
    db.commit()

def verify_api_key(
    api_key: str = Header(..., alias="Authorization"), db: Session = Depends(database.get_db)
):
    """
    Verifies provided api key against the database
    Additionally, checks if the is expired or revoked
    """
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record:
        expiration_check(api_key, db)
        if key_record.revoked:
            raise HTTPException(status_code=403, detail="API Key Revoked")
        if key_record.expired:
            raise HTTPException(status_code=403, detail="API Key Expired")
        return key_record
    raise HTTPException(status_code=403, detail="API Key Not Found")


def verify_api_key_with_rate_limit(
    request: Request,
    api_key: str = Header(..., alias="Authorization"),
    db: Session | None = Depends(database.get_db),
):
    """
    Verify API key AND enforce rate limiting.
    This is the new function that endpoints should use.
    """

    # Step 1: Verify API key (same as before)
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not key_record:
        raise HTTPException(status_code=403, detail="API Key Not Found")
    if key_record.revoked:
        raise HTTPException(status_code=403, detail="API Key Revoked")
    if key_record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=403, detail="API Key Expired")

    # Step 2: Check rate limit
    endpoint = request.url.path  # e.g., "/cells"
    if not check_rate_limit(api_key, endpoint, db):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Try again later."
        )

    # Step 3: Log this request (populates the table!)
    log_request(api_key, endpoint, db)

    # Step 4: Return key record (request proceeds)
    return key_record

def keys_disabled():
    return None

verify_api_key_func = verify_api_key if use_keys() else keys_disabled
verify_api_key_with_rate_limit_func = verify_api_key_with_rate_limit if use_keys() else keys_disabled


def generate_api_key():
    """
    Generates random base64-encoded url-secure string used as an api key
    """
    return secrets.token_urlsafe(32)

def expiration_check(api_key: str, db: Session):
    """
    Checks if the key has expired and changes key data accordingly
    """
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record.admin:
        return
    if key_record.expires_at <= datetime.utcnow():
        key_record.expired = True
        db.commit()
        db.refresh(key_record)

def verify_admin_api_key(api_key: str = Header(..., alias="Authorization"), db: Session = Depends(database.get_db)
):
    """
    Checks if provided key is an admin key
    """
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record:
        if not key_record.admin:
            raise HTTPException(status_code=403, detail="Key does not have admin rights")
        return key_record
    raise HTTPException(status_code=403, detail="API Key Not Found")

@keys_router.get("/public_stuff")
def access_public_data():
    """
    Testing function to check access to publicly available information
    """
    return "Public!"

@keys_router.get("/user_stuff")
def access_user_data(key=Depends(verify_api_key)):
    """
    Testing function to check access to information available with standard api key,
    requires user key to be accessed
    """
    return "User!"

@keys_router.get("/admin_stuff")
def access_admin_data(admin_key:str=Depends(verify_admin_api_key)):
    """
    Testing function to check access to information available with admin api key,
    requires admin key to be accessed
    """
    return "Admin!"


@keys_router.post(path="/create_user_api_key")
def create_user_api_key(db: Session = Depends(database.get_db)):
    """
    Generates a standart api key and saves it to the database
    """
    # generates api key
    key = generate_api_key()
    # calculates expiration datetime
    expiration = datetime.utcnow() + timedelta(days=settings.key_life_span)
    # creates new key object for database
    new_key_object = APIKey(key=key, expires_at=expiration)
    # puts the object in database
    db.add(new_key_object)
    # commits the changes to the database
    db.commit()
    db.refresh(new_key_object)
    return f"New user key generated: {key}, expires at: {expiration}"

@keys_router.post(path="/create_admin_api_key")
def create_admin_api_key(db: Session = Depends(database.get_db), admin_key:str=Depends(verify_admin_api_key)):
    """
    Generates an admin api key and saves it to the database,
     requires admin key to be accessed
    """
    key = generate_api_key()
    expiration = datetime.utcnow() + timedelta(days=settings.key_life_span)
    new_key_object = APIKey(key=key, expires_at=None, admin=True)
    db.add(new_key_object)
    db.commit()
    db.refresh(new_key_object)
    return f"New admin key generated: {key}, expires at: {expiration}"


@keys_router.get(path="/get_key_info/{key}")
def get_key_info(key: str, db: Session = Depends(database.get_db)):
    """
    Fetches information about the provided api key from the database
    """
    return db.query(APIKey).filter(APIKey.key == key).first()


@keys_router.put(path="/revoke_key/{key}")
def revoke_key(key: str, db: Session = Depends(database.get_db), admin_key: str = Depends(verify_admin_api_key)):
    """
    Revokes provided api key, requires admin key to be accessed
    """
    record = db.query(APIKey).filter(APIKey.key == key).first()
    if record:
        record.revoked = True
        db.commit()
        db.refresh(record)
        return "Key revoked successfully"
    return "Key doesn't exist"
