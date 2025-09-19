from fastapi import FastAPI

from views_challenge.api import api

app = FastAPI()
app.include_router(api.router)