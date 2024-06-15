import pandas as pd

df = pd.read_csv("data/tourism.csv")


def fetch_all_tourism():
    return df


def filter_tourism(name=None, city=None, categories=None):
    filtered_df = df.copy()
    if name:
        filtered_df = filtered_df[
            filtered_df["Place_Name"].str.contains(name, case=False, na=False)
        ]
    if city:
        filtered_df = filtered_df[
            filtered_df["City"].str.contains(city, case=False, na=False)
        ]
    if categories:
        # Convert categories to lowercase for case-insensitive comparison
        categories = [category.lower() for category in categories]
        filtered_df["Category"] = filtered_df["Category"].str.lower()

        # Filter by multiple categories
        category_filter = filtered_df["Category"].apply(
            lambda x: any(category in x for category in categories)
        )
        filtered_df = filtered_df[category_filter]
    return filtered_df
