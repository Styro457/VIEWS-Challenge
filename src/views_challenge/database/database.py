"""
Objects and constants necessary to use Postgres database
"""


# engine ~ used by alchemy, talks to postgres ~ connection
from sqlalchemy import create_engine
# factory used to create session objects
from sqlalchemy.orm import sessionmaker

# dialect://username:password@host:port/database
# should be stored in .env
DATABASE_USER="postgres"
DATABASE_PASSWORD="admin"
DATABASE_PORT="5432"
DB_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@localhost:{DATABASE_PORT}/views_api"
print(DB_URL)
# initiates the engine, it knows how to talk to and manage the db
engine = create_engine(DB_URL)
# creates session factory binded to engine
SessionLocal = sessionmaker(bind=engine)
