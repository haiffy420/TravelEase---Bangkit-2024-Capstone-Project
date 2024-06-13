---
title: Travelease
emoji: 📉
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# TravelEase - Machine Learning Documentation

Welcome to the TravelEase Machine Learning Documentation. TravelEase is a mobile application that provides a tourist attraction recommendation system based on user preferences. This document provides a comprehensive guide on the machine learning component of the TravelEase project.

## Table of Contents

- [TravelEase - Machine Learning Documentation](#travelease---machine-learning-documentation)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Machine Learning Team](#machine-learning-team)
    - [Team ID : C241-PS237](#team-id--c241-ps237)
  - [Data Wrangling](#data-wrangling)
    - [Gathering Data](#gathering-data)
    - [Assessing and Cleaning Data](#assessing-and-cleaning-data)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Data Summary](#data-summary)
    - [Visualizations](#visualizations)
  - [Data Modeling](#data-modeling)
    - [TF-IDF Vectorizer](#tf-idf-vectorizer)
    - [Model Training](#model-training)
    - [Recommendation Function](#recommendation-function)
  - [Model Deployment](#model-deployment)
    - [FastAPI Application](#fastapi-application)
    - [Recommender Function](#recommender-function)
    - [Dockerfile](#dockerfile)
  - [How to Run](#how-to-run)
    - [Local](#local)
    - [Docker](#docker)
  - [API Endpoints](#api-endpoints)
      - [Request body](#request-body)
      - [Example Value](#example-value)
      - [Responses](#responses)
  - [Dependencies](#dependencies)
  - [Credits](#credits)
    - [About the Dataset](#about-the-dataset)

## Project Overview

TravelEase's primary features include:
- A recommendation system for tourist destinations.
- Auto-generated itineraries.
- Trip cost tracking.
- Tourist reviews.

This documentation focuses on the machine learning part of the project, which involves creating a recommendation system for tourist destinations.

## Machine Learning Team

### Team ID : C241-PS237

<br>

| Name                              | Student ID    | Path                |
| --------------------------------- | ------------- | ------------------- |
| Haifan Tri Buwono Joyo Pangestu   | M693D4KY2338  | Machine Learning    |
| Nisrina Diva Sulalah              | M006D4KX2082  | Machine Learning    |
| Ariqa Bilqis                      | M006D4KX2081  | Machine Learning    |

<br>

## Data Wrangling

### Gathering Data
We use a dataset containing information about various tourist destinations.

```python
import pandas as pd

PATH = 'https://raw.githubusercontent.com/haiffy420/TravelEase---Bangkit-2024-Capstone-Project/main/data'
tourism = pd.read_csv(f"{PATH}/tourism_with_id.csv")
```

### Assessing and Cleaning Data
We assessed and cleaned the data to ensure it's ready for analysis and modeling.

```python
# Checking for missing values
tourism.info()
tourism.isnull().sum()

# Filling missing values
tourism['Time_Minutes'] = tourism['Time_Minutes'].fillna(60)

# Dropping unused columns
tourism = tourism.drop(columns=['Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'])

# Checking for duplicates
print("Number of data duplications (tourism): ", tourism.duplicated().sum())
```

## Exploratory Data Analysis (EDA)

We performed EDA to understand the distribution and relationships within the data.

### Data Summary
```python
print(tourism.describe(include='all'))
```

### Visualizations
We created several visualizations to explore the data.

- **Category Distribution**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

category_counts = df['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Frequency']
sns.barplot(data=category_counts, x='Category', y='Frequency', palette='viridis')
plt.title('Category Distribution')
plt.show()
```

- **Rating Distribution**:
```python
rating_counts = df['Rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Frequency']
sns.barplot(data=rating_counts, x='Rating', y='Frequency', palette='viridis')
plt.title('Rating Distribution')
plt.show()
```

- **Most Popular Tourist Destinations**:
```python
top_10 = df.sort_values(by='Rating', ascending=False).head(10)
sns.barplot(x='Rating', y='Place_Name', data=top_10, palette='viridis')
plt.title('Top 10 Tourist Destinations Based on Highest Rating')
plt.show()
```

## Data Modeling

### TF-IDF Vectorizer
We used TF-IDF to vectorize the categories of tourist destinations.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Category'])
```

### Model Training
We built and trained a neural network model to predict tourist destinations based on categories.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def build_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(input_dim, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(tfidf_matrix.shape[1])
X_train, X_test = train_test_split(tfidf_matrix.toarray(), test_size=0.2, random_state=42)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90 and logs.get('val_accuracy') > 0.90:
            self.model.stop_training = True

model.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_test, X_test), callbacks=[myCallback()])
```

### Recommendation Function
The recommendation function suggests places based on user input categories and city.

```python
def place_recommendations(categories, city, items=df[['Category', 'Place_Name', 'Rating', 'City']], k=5):
    city_items = items[items['City'] == city]
    if len(items) < k:
        k = len(items)
    input_vector = vectorizer.transform(categories).toarray()
    predicted_vector = model.predict(input_vector)
    city_tfidf_matrix = vectorizer.transform(city_items['Category']).toarray()
    city_similarities = cosine_similarity(predicted_vector, city_tfidf_matrix)
    similar_indices = np.argsort(city_similarities, axis=1)[:, ::-1][:, :k]
    recommendations = pd.DataFrame(columns=['Place_Name', 'Category', 'Rating', 'City'])
    for i, indices in enumerate(similar_indices):
        category_places = city_items.iloc[indices]
        category_places['Category'] = categories[i]
        recommendations = pd.concat([recommendations, category_places])
    recommendations = recommendations.sort_values(by='Rating', ascending=False)
    return recommendations
```

## Model Deployment

We deployed the model using FastAPI.

### FastAPI Application

```python
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
```

### Recommender Function

```python
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("data/tourism.csv")
model = tf.keras.models.load_model("model/travelease.h5")
vectorizer = TfidfVectorizer()
vectorizer.fit(df["Category"])

def recommender(categories, city, items=df[["Place_Id", "Place_Name", "Description", "Category", "City", "Price", "Rating", "Time_Minutes", "Coordinate"]], k=5):
    city_items = items[items["City"] == city]
    if len(city_items) < k:
        k = len(city_items)
    input_vector = vectorizer.transform(categories).toarray()
    predicted_vector = model.predict(input_vector)
    city_tfidf_matrix = vectorizer.transform(city_items["Category"]).toarray()
    city_similarities = cosine_similarity(predicted_vector, city_tfidf_matrix)
    similar_indices = np.argsort(city_similarities, axis=1)[:, ::-1][:, :k]
    recommendations = pd.DataFrame(columns=["Place_Id", "Place_Name", "Description", "Category", "City", "Price", "Rating", "Time_Minutes", "Coordinate"])
    for i, indices in enumerate(similar_indices):
        category_places = city_items.iloc[indices]
        category_places["Category"] = categories[i]
        recommendations = pd.concat([recommendations, category_places])
    recommendations = recommendations.sort_values(by="Rating", ascending=False)
    return recommendations
```

### Dockerfile

```dockerfile
FROM python:3.9

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## How to Run
### Local
1. Clone the repository:
    ```bash
	git clone https://github.com/haiffy420/TravelEase---Bangkit-2024-Capstone-Project.git
    cd TravelEase---Bangkit-2024-Capstone-Project/travelease-deployment
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 7860
    ```
### Docker
1. Clone the repository:
    ```bash
	git clone https://github.com/haiffy420/TravelEase---Bangkit-2024-Capstone-Project.git
    cd TravelEase---Bangkit-2024-Capstone-Project/travelease-deployment
    ```
2. Build and run the Docker container:
    ```bash
    docker build -t travelease .
    docker run -p 7860:7860 travelease
    ```
## API Endpoints

- **GET /**: Returns a welcome message.

- **POST /recommend/**: Returns recommended tourist destinations based on the input categories and city.
#### Request body
`application/json`
#### Example Value

```json
{
  "categories": [
    "string"
  ],
  "city": "string"
}
```

#### Responses

Code: 200
Description: Successful Response
Media type: `application/json`
-   Example Value

```json
"string"
```
Code: 422
Description: Validation Error
Media type: `application/json`
-   Example Value

```json
{
  "detail": [
    {
      "loc": [
        "string",
        0
      ],
      "msg": "string",
      "type": "string"
    }
  ]
}
```

## Dependencies

- Python 3.9
- Pandas
- TensorFlow
- Scikit-learn
- FastAPI
- Uvicorn

## Credits

This project uses the dataset [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) from Kaggle by the [GetLoc Team](https://github.com/AgungP88/getloc-apps). 

### About the Dataset
- **Problem**: Before traveling, usually someone will make a plan in advance about the location to be visited and the time of departure. This is done to avoid problems, one of which is the distance to be traveled and the time needed does not match expectations.
- **Content**: This dataset contains several tourist attractions in 5 major cities in Indonesia: Jakarta, Yogyakarta, Semarang, Bandung, and Surabaya. It was used in the Capstone Project Bangkit Academy 2021 called GetLoc, an application that recommends tourist destinations based on user preferences, city, price, category, and time.
