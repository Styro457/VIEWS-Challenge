"""
SQLAlchemy ORM Models for Postgres database
"""


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


# represents the table for tracking API requests for rate limiting
class RequestLog(Base):
    __tablename__ = "request_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key = Column(String, ForeignKey("api_keys.key"), nullable=False)
    endpoint = Column(String, nullable=False)  # which endpoint was called
    timestamp = Column(DateTime, nullable=False)  # when the request was made
    

