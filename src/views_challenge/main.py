from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware
from views_challenge.api import api

app = FastAPI()
app.include_router(api.router)

app.add_middleware(GZipMiddleware, minimum_size=1024)

