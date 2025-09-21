"""
Objects and constants necessary to use Postgres database
"""
from typing import Optional

# engine ~ used by alchemy, talks to postgres ~ connection
from sqlalchemy import create_engine
# factory used to create session objects
from sqlalchemy.orm import sessionmaker, Session

from views_challenge.configs.config import settings

class Database:

    def __init__(self):
        self.engine = None
        self.session_local = None

    def connect(self, database_url : str):
        #TODO: Implement proper logging
        print(f"Connecting to database ({database_url})")

        # Initiate engine
        self.engine = create_engine(database_url)

        # Create session factory binded to engine
        self.session_local = sessionmaker(bind=self.engine)

    def get_db(self):
        if self.engine is None:
            return None
        db = self.session_local()
        try:
            # yields the session for the function that called get_db()
            yield db
        # cleanup after the session is not needed anymore
        finally:
            # closes the session
            db.close()

database = Database()
if settings.database_user is not None and settings.database_user != "":
    DB_URL = f"postgresql://{settings.database_user}:{settings.database_password}@localhost:{settings.database_port}/views_api"
    print(DB_URL)
    database.connect(DB_URL)
