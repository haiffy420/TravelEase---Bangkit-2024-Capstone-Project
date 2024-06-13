from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from recommender import recommender

app = FastAPI()

class Item(BaseModel):
    categories: list
    city: str
    
@app.post("/")
def hello():
    return {"message": "TravelEase - All-in-One Trip Companion"}

@app.post("/recommend/")
async def recommend_places(item: Item):
    categories = item.categories
    city = item.city
    recommendations = recommender(categories, city)
    return recommendations.to_dict(orient="records")
