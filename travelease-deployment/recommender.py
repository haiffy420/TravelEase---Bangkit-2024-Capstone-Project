import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
df = pd.read_csv("data/tourism.csv")

# Load pre-trained model
model = tf.keras.models.load_model("model/travelease.h5")

# Initialize and fit the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df["Category"])


def recommender(
    categories,
    city,
    items=df[
        [
            "Place_Id",
            "Place_Name",
            "Description",
            "Category",
            "City",
            "Price",
            "Rating",
            "Time_Minutes",
            "Coordinate",
        ]
    ],
    k=5,
):
    # Filter dataframe based on specified city
    city_items = items[items["City"].str.lower() == city.lower()]

    # If the number of items after filtering is less than k, adjust k
    if len(city_items) < k:
        k = len(city_items)

    # Transform input categories to TF-IDF vectors
    input_vector = vectorizer.transform(categories).toarray()

    # Predict TF-IDF representation of input categories using the trained model
    predicted_vector = model.predict(input_vector)

    # Compute cosine similarity between predicted vector and TF-IDF matrix for the city's items
    city_tfidf_matrix = vectorizer.transform(city_items["Category"]).toarray()
    city_similarities = cosine_similarity(predicted_vector, city_tfidf_matrix)

    # Get indices of top similar places
    similar_indices = np.argsort(city_similarities, axis=1)[:, ::-1][:, :k]

    # Initialize an empty DataFrame to collect recommendations
    recommendations = pd.DataFrame(
        columns=[
            "Place_Id",
            "Place_Name",
            "Description",
            "Category",
            "City",
            "Price",
            "Rating",
            "Time_Minutes",
            "Coordinate",
        ]
    )

    # Collect recommended places
    for i, indices in enumerate(similar_indices):
        category_places = city_items.iloc[indices]
        category_places["Category"] = categories[
            i
        ]  # Assign the corresponding category to the recommended places
        recommendations = pd.concat([recommendations, category_places])

    # Sort recommendations based on ratings
    recommendations = recommendations.sort_values(by="Rating", ascending=False)

    return recommendations
