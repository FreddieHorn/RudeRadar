import uvicorn
from fastapi import FastAPI
from inference import RudenessDeterminator

app = FastAPI()
rudness_determinator = RudenessDeterminator()

@app.get("/")
async def root():
    return {"message": "This is a microservice responsible for inference on the trained model, which determines rudeness level of the queried text"}

@app.get("/model")
async def get_rudeness_level(text: str):
    score = rudness_determinator.measure_rudeness(text)
    return {"score" : score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
