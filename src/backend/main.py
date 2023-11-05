import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import the CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import os
from bson.json_util import dumps

from model.inference import RudenessDeterminator

class Item(BaseModel):
    text: str

# Use environment variables to get MongoDB connection details
mongo_host = os.environ.get("MONGO_HOST", "mongodb-0")
mongo_port = int(os.environ.get("MONGO_PORT", 27017))
db_name = os.environ.get("MONGO_DB_NAME", "db1")
collection_name = os.environ.get("MONGO_COLLECTION_NAME", "collection1")

app = FastAPI()
rudness_determinator = RudenessDeterminator()

# Add CORS Middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the specific origins you want to allow
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up the MongoDB connection using the environment variables
mongo_client = MongoClient(host=mongo_host, port=mongo_port)
db = mongo_client[db_name]
collection = db[collection_name]

@app.get("/")
async def root():
    return {"message": "This is a microservice responsible for inference on the trained model, which determines rudeness level of the queried text"}

@app.post("/model_predict")
async def get_rudeness_level(item: Item):  
    return {"score" : rudness_determinator.measure_rudeness(item.text)}

@app.get("/read_data/")
async def read_data():
    cursor = collection.find({})    
    data = list(cursor)
    return {"data": dumps(data)}

# Create an endpoint to send data to MongoDB
@app.post("/send_data/")
async def send_data(text: str, score: int):
    data = {"text": text, "score": score}
    inserted_data = collection.insert_one(data)
    return {"message": "Data sent to MongoDB", "inserted_id": str(inserted_data.inserted_id)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
