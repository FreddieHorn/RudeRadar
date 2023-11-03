import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import the CORSMiddleware
from pydantic import BaseModel

from model.inference import RudenessDeterminator

class Item(BaseModel):
    text: str


app = FastAPI()
rudness_determinator = RudenessDeterminator()

# Add CORS Middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with the specific origins you want to allow
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "This is a microservice responsible for inference on the trained model, which determines rudeness level of the queried text"}

@app.post("/model_predict")
async def get_rudeness_level(item: Item):   
    return {"score" : rudness_determinator.measure_rudeness(item.text)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
