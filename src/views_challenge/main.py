"""
Program entry point
"""
from fastapi import FastAPI
from views_challenge.database.database import database
from views_challenge.database.models import Base
from starlette.middleware.gzip import GZipMiddleware

from views_challenge.api import keys_handler
from views_challenge.api import api
from views_challenge.configs.config import settings

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=settings.response_compression_min)
app.include_router(api.router)


if database.is_connected():
    app.include_router(keys_handler.keys_router, prefix="/auth")

    Base.metadata.create_all(bind=database.engine)
    #
    # @app.on_event("startup")
    # def on_startup():
    #    """Called on API startup, updates database schemas"""
    #    Base.metadata.create_all(bind=engine)
   
