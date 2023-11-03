import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import the CORSMiddleware
from pydantic import BaseModel
import requests
import json

class Item(BaseModel):
    text: str

app = FastAPI()
model_URL = "http://localhost:8000/model_predict"

# Add CORS Middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the specific origins you want to allow
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/divert_to_model")
async def get_rudeness_level(item: Item):
    return requests.post(model_URL, json = {"text" : item.text}).json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)