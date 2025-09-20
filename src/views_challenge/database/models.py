from sqlalchemy import Column, String, Boolean, DateTime, Integer

# function that creates a base class for ORM models
from sqlalchemy.ext.declarative import declarative_base

# initiates base class for all models
Base = declarative_base()

DEFAULT_DAILY_REQUESTS_LIMIT = 1000

# represents the table for keys
class APIKey(Base):
    """ORM Model for storing data about api keys"""
    __tablename__ = "api_keys"
    key = Column(String, primary_key=True)
    expires_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False)
    daily_limit = Column(Integer, default=DEFAULT_DAILY_REQUESTS_LIMIT)
    admin = Column(Boolean, default=False)
    expired = Column(Boolean, default=False)
