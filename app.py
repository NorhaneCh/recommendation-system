import joblib
import pandas as pd
import streamlit as st

df = pd.read_csv('ecommerce_data.csv', encoding='latin1')
# List of items to display in the dropdown
items = df['Description']

# Load the knn model
knn = joblib.load('knn.joblib')

# Load the tfidf_matrix
tfidf_matrix = joblib.load('tfidf_matrix.joblib')


# Assuming there's a column named 'Description' for item names
item_id_to_name = dict(zip(df.index, df['Description']))
item_name_to_id = dict(zip(df['Description'], df.index))

# Function to recommend similar items


def recommend_similar_items(item_name, num_recommendations):

    item_index = item_name_to_id.get(item_name)
    item_vector = tfidf_matrix[item_index]

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(
        item_vector, n_neighbors=num_recommendations + 1)
    similar_indices = indices.flatten()
    return similar_indices


# Function to get similar items by item name
def get_similar_items_by_name(item_name):
    similar_items_ids = recommend_similar_items(item_name, 10)
    similar_items_names = [item_id_to_name.get(
        item) for item in similar_items_ids]
    similar_items_names = [
        item for item in similar_items_names if item != item_name]
    return similar_items_names


st.title("E-Commerce Recommendation System")

# List of items to display in the dropdown
items = df['Description']

# Creating the dropdown list
selected_item = st.selectbox('Choose an item :', items)
recommendations = get_similar_items_by_name(selected_item)
st.write("Recommended items are:")
st.write(recommendations)
