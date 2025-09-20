# cryptography module for token generation
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from views_challenge.database.database import SessionLocal
from views_challenge.database.models import APIKey

router = APIRouter()
print("API router loaded")

# key life span in days
KEY_LIFE_SPAN = 90


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
