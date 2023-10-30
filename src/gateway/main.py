import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import the CORSMiddleware
from pydantic import BaseModel
import requests

class Item(BaseModel):
    text: str

app = FastAPI()
model_URL = "http://localhost:8000/model_predict"

@app.post("/divert_to_model")
async def get_rudeness_level(item: Item):
    return requests.post(model_URL, json = Item)
