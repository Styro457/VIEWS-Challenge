# cryptography module for token generation
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session
from views_challenge.database.database import SessionLocal
from views_challenge.database.models import APIKey, RequestLog

router = APIRouter()
print("API router loaded")

# key life span in days
KEY_LIFE_SPAN = 90

# rate limiting settings
RATE_LIMIT_REQUESTS = 100  # requests per hour
RATE_LIMIT_WINDOW = 60  # minutes


def get_db():
    # opening db connection for current request
    db = SessionLocal()
    try:
        # yields the session for the function that called get_db()
        yield db
    # cleanup after the session is not needed anymore
    finally:
        # closes the session
        db.close()


def check_rate_limit(api_key: str, endpoint: str, db: Session) -> bool:
    """
    Check if the API key has exceeded the rate limit.
    Returns True if within limit, False if exceeded.
    """
    # Calculate the time window (now - RATE_LIMIT_WINDOW minutes)
    window_start = datetime.utcnow() - timedelta(minutes=RATE_LIMIT_WINDOW)

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
    return request_count < RATE_LIMIT_REQUESTS


def log_request(api_key: str, endpoint: str, db: Session):
    """Log the current request for rate limiting."""
    request_log = RequestLog(
        api_key=api_key, endpoint=endpoint, timestamp=datetime.utcnow()
    )
    db.add(request_log)
    db.commit()


def verify_api_key(
    api_key: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)
):
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record:
        if key_record.revoked:
            raise HTTPException(status_code=403, detail="API Key Revoked")
        if key_record.expires_at < datetime.utcnow():
            raise HTTPException(status_code=403, detail="API Key Expired")
        return key_record
    raise HTTPException(status_code=403, detail="API Key Not Found")


def verify_api_key_with_rate_limit(
    request: Request,
    api_key: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_db),
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


def generate_api_key():
    return secrets.token_urlsafe(32)


@router.get("/protected_stuff")
def access_protected_data(key=Depends(verify_api_key)):
    return "Success!"


@router.post(path="/create_api_key")
def create_api_key(db: Session = Depends(get_db)):
    # generates api key
    key = generate_api_key()
    # calculates expiration datetime
    expiration = datetime.utcnow() + timedelta(days=KEY_LIFE_SPAN)
    # creates new key object for database
    new_key_object = APIKey(key=key, expires_at=expiration)
    # puts the object in database
    db.add(new_key_object)
    # commits the changes to the database
    db.commit()
    db.refresh(new_key_object)
    return f"New key generated: {key}, expires at: {expiration}"


@router.get(path="/get_key_info/{key}")
def get_key_info(key: str, db: Session = Depends(get_db)):
    return db.query(APIKey).filter(APIKey.key == key).first()


@router.put(path="/revoke_key/{key}")
def revoke_key(key: str, db: Session = Depends(get_db)):
    record = db.query(APIKey).filter(APIKey.key == key).first()
    if record:
        record.revoked = True
        db.commit()
        db.refresh(record)
        return "Key revoked successfully"
    return "Key doesn't exist"
