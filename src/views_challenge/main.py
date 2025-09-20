import views_challenge.api.api as api
from fastapi import FastAPI
from views_challenge.database.database import engine
from views_challenge.data.models import Base
from starlette.middleware.gzip import GZipMiddleware
from views_challenge.api import api

app = FastAPI()
app.include_router(api.router)

app.add_middleware(GZipMiddleware, minimum_size=1024)

#Base.metadata.create_all(bind=engine)

#@app.on_event("startup")
#def on_startup():
#    Base.metadata.create_all(bind=engine)
   