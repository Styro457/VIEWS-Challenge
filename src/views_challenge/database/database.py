# engine ~ used by alchemy, talks to postgres ~ connection
from sqlalchemy import create_engine

# factory used to create session objects
from sqlalchemy.orm import sessionmaker

# dialect://username:password@host:port/database
# should be stored in .env
DB_URL = "postgresql://postgres:IHatePostgres@localhost:5432/views_api"
# initiates the engine, it knows how to talk to and manage the db
engine = create_engine(DB_URL)
# creates session factory binded to engine
SessionLocal = sessionmaker(bind=engine)
