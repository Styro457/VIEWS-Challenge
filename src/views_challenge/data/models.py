from sqlalchemy import Column, String, Boolean, DateTime, Integer
# function that creates a base class for ORM models
from sqlalchemy.ext.declarative import declarative_base

# initiates base class for all models
Base = declarative_base()

# represents the table for keys
class APIKey(Base):
    __tablename__ = "api_keys"
    key = Column(String, primary_key=True)
    expires_at = Column(DateTime)
    revoked  = Column(Boolean, default=False)

