import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Ensure this import is present

# Page Configuration
st.set_page_config(page_title="Library Recommendation System", layout="wide")

# Title
st.title("Smart Library Recommendation System")

# Sidebar for navigation
st.sidebar.title("Modules")
module = st.sidebar.selectbox(
    "Select a Module",
    ["Data Cleaning", "Recommendation System"]
)

# Upload dataset
@st.cache_data
def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

uploaded_file = st.sidebar.file_uploader("Upload Your Dataset (CSV)", type=["csv"])

if uploaded_file:
    user_data = load_dataset(uploaded_file)
else:
    # Use default data if no file is uploaded
    user_data = pd.DataFrame({
        "Book_ID": [1, 2, 3, 4],
        "Title": ["Book A", "Book B", "Book C", "Book D"],
        "Author": ["Author A", "Author B", "Author C", None],
        "Genre": ["Fiction", "Non-Fiction", "Sci-Fi", "Fiction"],
        "ISBN": ["9780131103627", "9780201633610", None, "9780131103627"],
        "Year": [1995, 2000, None, 1995]
    })

# Data Cleaning Module
if module == "Data Cleaning":
    st.header("Data Cleaning Module")

    # Display Original Data
    st.subheader("Original Dataset")
    st.write(user_data)

    # Handle Missing Data
    st.subheader("Handling Missing Data")
    cleaned_data = user_data.replace({None: np.nan})
    imputer = SimpleImputer(strategy="most_frequent")
    cleaned_data.iloc[:, :] = imputer.fit_transform(cleaned_data)

    # Duplicate Removal
    st.subheader("Removing Duplicates")
    cleaned_data = cleaned_data.drop_duplicates(subset="ISBN")

    # Display Cleaned Data
    st.subheader("Cleaned Dataset")
    st.write(cleaned_data)

    # Persist cleaned data for Recommendation System
    st.session_state["cleaned_data"] = cleaned_data

# Recommendation System Module
elif module == "Recommendation System":
    st.header("Recommendation System Module")

    # Ensure cleaned data is available
    if "cleaned_data" in st.session_state:
        user_data = st.session_state["cleaned_data"]
    else:
        st.error("Please complete the Data Cleaning step first.")
        st.stop()

    st.subheader("Select Genre for Book Recommendations")

    # Dynamically get unique genres from the dataset
    if "Genre" in user_data.columns:
        unique_genres = user_data["Genre"].dropna().unique()
        selected_genre = st.selectbox("Choose a Genre", options=unique_genres)

        # Filter books by the selected genre
        filtered_books = user_data[user_data["Genre"] == selected_genre]

        if not filtered_books.empty:
            st.subheader(f"Books in Genre: {selected_genre}")
            for idx, row in filtered_books.iterrows():
                # Handle missing or absent columns
                title = row["Title"] if "Title" in row else "Unknown Title"
                author = row["Author"] if "Author" in row and pd.notna(row["Author"]) else "Unknown Author"
                year = row["Year"] if "Year" in row and pd.notna(row["Year"]) else "Unknown Year"
                isbn = row["ISBN"] if "ISBN" in row and pd.notna(row["ISBN"]) else "Unknown ISBN"

                # Display book details
                st.markdown("---")
                st.subheader(title)
                st.write(f"**Author(s):** {author}")
                st.write(f"**Published Year:** {year}")
                st.write(f"**ISBN:** {isbn}")
                st.write(f"**Genre:** {selected_genre}")
            st.markdown("---")
        else:
            st.warning("No books available for the selected genre.")
    else:
        st.error("The dataset does not contain a 'Genre' column.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Expert Systems Final Exam")
