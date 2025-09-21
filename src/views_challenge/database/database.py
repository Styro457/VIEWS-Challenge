"""
Objects and constants necessary to use Postgres database
"""


# engine ~ used by alchemy, talks to postgres ~ connection
from sqlalchemy import create_engine
# factory used to create session objects
from sqlalchemy.orm import sessionmaker
from views_challenge.configs.config import settings

# dialect://username:password@host:port/database
# should be stored in .env
DB_URL = f"postgresql://{settings.database_user}:{settings.database_password}@localhost:{settings.database_port}/views_api"
print(DB_URL)

# initiates the engine, it knows how to talk to and manage the db
engine = create_engine(DB_URL)
# creates session factory binded to engine
SessionLocal = sessionmaker(bind=engine)
