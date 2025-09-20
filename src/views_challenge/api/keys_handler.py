# cryptography module for token generation
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from views_challenge.database.database import SessionLocal
from views_challenge.database.models import APIKey

keys_router = APIRouter()


# key life span in days
KEY_LIFE_SPAN = 90
KEYS_MODE = True


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
        expiration_check(api_key, db)
        if key_record.revoked:
            raise HTTPException(status_code=403, detail="API Key Revoked")
        if key_record.expired:
            raise HTTPException(status_code=403, detail="API Key Expired")
        return key_record
    raise HTTPException(status_code=403, detail="API Key Not Found")


def generate_api_key():
    return secrets.token_urlsafe(32)

def expiration_check(api_key: str, db: Session):
    """Checks if the key status should be changed to expired"""
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record.admin:
        return
    if key_record.expires_at <= datetime.utcnow():
        key_record.expired = True
        db.commit()
        db.refresh(key_record)

def verify_admin_api_key(api_key: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)
):
    """Checks if provided key is an admin key"""
    key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if key_record:
        if not key_record.admin:
            raise HTTPException(status_code=403, detail="Key does not have admin rights")
        return key_record
    raise HTTPException(status_code=403, detail="API Key Not Found")

@keys_router.get("/public_stuff")
def access_public_data():
    return "Public!"

@keys_router.get("/user_stuff")
def access_user_data(key=Depends(verify_api_key)):
    return "User!"

@keys_router.get("/admin_stuff")
def access_admin_data(admin_key:str=Depends(verify_admin_api_key)):
    return "Admin!"


@keys_router.post(path="/create_user_api_key")
def create_user_api_key(db: Session = Depends(get_db)):
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
    return f"New user key generated: {key}, expires at: {expiration}"

@keys_router.post(path="/create_admin_api_key")
def create_admin_api_key(db: Session = Depends(get_db), admin_key:str=Depends(verify_admin_api_key)):
    key = generate_api_key()
    expiration = datetime.utcnow() + timedelta(days=KEY_LIFE_SPAN)
    new_key_object = APIKey(key=key, expires_at=None, admin=True)
    db.add(new_key_object)
    db.commit()
    db.refresh(new_key_object)
    return f"New admin key generated: {key}, expires at: {expiration}"


@keys_router.get(path="/get_key_info/{key}")
def get_key_info(key: str, db: Session = Depends(get_db)):
    return db.query(APIKey).filter(APIKey.key == key).first()


@keys_router.put(path="/revoke_key/{key}")
def revoke_key(key: str, db: Session = Depends(get_db), admin_key: str = Depends(verify_admin_api_key)):
    record = db.query(APIKey).filter(APIKey.key == key).first()
    if record:
        record.revoked = True
        db.commit()
        db.refresh(record)
        return "Key revoked successfully"
    return "Key doesn't exist"
