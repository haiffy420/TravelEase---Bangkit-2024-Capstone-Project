import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from recommender import recommender
from data import fetch_all_tourism, filter_tourism

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

@app.get("/tourism/")
async def get_tourism_data(
    name: Optional[str] = Query(None, description="Filter by place name"),
    city: Optional[str] = Query(None, description="Filter by city"),
    categories: Optional[List[str]] = Query(None, description="Filter by categories"),
):
    print(name, city, categories)
    if not any([name, city, categories]):
        data = fetch_all_tourism()
    else:
        data = filter_tourism(name, city, categories)
    return data.to_dict(orient="records")
