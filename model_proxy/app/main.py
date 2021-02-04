from fastapi import FastAPI

from app.api import router_ping, router_predictions, router_statistics
from app.db import database, engine, metadata

metadata.create_all(engine)

app = FastAPI()


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


app.include_router(router_ping.router)
app.include_router(router_predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(router_statistics.router, prefix="/statistics", tags=["statistics"])
