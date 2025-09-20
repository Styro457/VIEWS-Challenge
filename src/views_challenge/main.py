import views_challenge.api.api as api
from fastapi import FastAPI
from views_challenge.database.database import engine
from views_challenge.data.models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(api.router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
