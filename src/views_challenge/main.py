import views_challenge.api.api as api
from fastapi import FastAPI
from views_challenge.database.database import engine
from views_challenge.database.models import Base
from starlette.middleware.gzip import GZipMiddleware
from views_challenge.api import api
from views_challenge.configs.config import settings

app = FastAPI()
app.include_router(api.router)

app.add_middleware(GZipMiddleware, minimum_size=settings.response_compression_min)

#Base.metadata.create_all(bind=engine)

#@app.on_event("startup")
#def on_startup():
#    Base.metadata.create_all(bind=engine)
   