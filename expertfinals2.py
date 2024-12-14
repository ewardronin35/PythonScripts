# library_recommender.py
from sqlalchemy import func

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk  # Requires Pillow library
import requests
from io import BytesIO
from sqlalchemy.orm import joinedload
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from pyod.models.iforest import IForest
import os
from datetime import datetime
import bcrypt
import threading  # For running initialization in a separate thread
import queue  # For thread-safe GUI updates
import logging  # For logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import concurrent.futures

# Use TkAgg backend for matplotlib
matplotlib.use("TkAgg")

# ------------------------------
# Configure Logging
# ------------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("library_recommender.log"),
                        logging.StreamHandler()
                    ])

# ------------------------------
# Image Paths from Local Directory
# ------------------------------

# Ensure that '1.png' exists in the same directory as this script
IMAGE_PATH = '1.png'  # Replace with your actual image file name if different

# Assign the same image to all roles as per user request
LOGO_PATH = IMAGE_PATH
LOGIN_ICON_PATH = IMAGE_PATH
REGISTER_ICON_PATH = IMAGE_PATH
HEADER_ICON_PATH = IMAGE_PATH

# ------------------------------
# Utility Function to Load Images from Local Files
# ------------------------------

def load_image(path, size):
    """
    Loads an image from the specified local path and resizes it to the given size.

    Args:
        path (str): The file path of the image.
        size (tuple): Desired size as (width, height).

    Returns:
        ImageTk.PhotoImage or None: The loaded image or None if failed.
    """
    if not os.path.exists(path):
        logging.warning(f"Image file '{path}' not found.")
        return None
    try:
        image = Image.open(path)
        # Use Image.Resampling.LANCZOS if available (Pillow >=9.1.0)
        if hasattr(Image, 'Resampling'):
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.ANTIALIAS  # For older Pillow versions
        image = image.resize(size, resample=resample_method)
        return ImageTk.PhotoImage(image)
    except Exception as e:
        logging.warning(f"Failed to load image from '{path}': {e}")
        return None

# ------------------------------
# Database Setup
# ------------------------------

Base = declarative_base()


class Book(Base):
    __tablename__ = 'books'
    book_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    genre = Column(String(100), nullable=False)
    ISBN = Column(String(20), unique=True, nullable=False)
    publication_year = Column(Integer, nullable=False)
    summary = Column(String)
    keywords = Column(String)
    borrow_count = Column(Integer, default=0)
    ratings = relationship("Rating", back_populates="book")
    borrowed = relationship("BorrowedBook", back_populates="book")


class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)  # Hashed password
    registration_date = Column(Date)
    ratings = relationship("Rating", back_populates="user")
    borrowed = relationship("BorrowedBook", back_populates="user")


class BorrowedBook(Base):
    __tablename__ = 'borrowed_books'
    borrow_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    book_id = Column(Integer, ForeignKey('books.book_id'))
    borrow_date = Column(Date)
    return_date = Column(Date)
    user = relationship("User", back_populates="borrowed")
    book = relationship("Book", back_populates="borrowed")


class Rating(Base):
    __tablename__ = 'ratings'
    rating_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    book_id = Column(Integer, ForeignKey('books.book_id'))
    rating = Column(Float)
    rating_date = Column(Date)
    user = relationship("User", back_populates="ratings")
    book = relationship("Book", back_populates="ratings")


def setup_database(db_url='mysql+pymysql://root:@localhost/library'):
    """
    Sets up the database connection and creates tables if they don't exist.

    Args:
        db_url (str): The database connection URL.

    Returns:
        Session or None: SQLAlchemy session object or None if failed.
    """
    try:
        engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        logging.info("Database setup completed.")
        return Session()
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        return None

# ------------------------------
# User Authentication Functions
# ------------------------------

def hash_password(password):
    """
    Hashes a plaintext password using bcrypt.

    Args:
        password (str): The plaintext password.

    Returns:
        bytes: The hashed password.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def verify_password(password, hashed):
    """
    Verifies a plaintext password against a hashed password.

    Args:
        password (str): The plaintext password.
        hashed (str): The hashed password.

    Returns:
        bool: True if passwords match, False otherwise.
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ------------------------------
# Data Cleaning Functions
# ------------------------------

def remove_duplicates(df, subset=['ISBN']):
    initial_count = df.shape[0]
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    final_count = df_cleaned.shape[0]
    message = f"Removed {initial_count - final_count} duplicate records."
    logging.info(message)
    return message, df_cleaned


def validate_isbn(isbn):
    if pd.isna(isbn):
        return False
    isbn_str = str(isbn)
    return (len(isbn_str) == 10 or len(isbn_str) == 13) and isbn_str.isdigit()


def fetch_book_metadata(isbn, message_queue=None):
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        key = f"ISBN:{isbn}"
        if key in data:
            book_data = data[key]
            authors = ", ".join([author['name'] for author in book_data.get('authors', [])])
            genres = ", ".join([subject['name'] for subject in book_data.get('subjects', [])])
            summary = book_data.get('notes', '')
            if message_queue:
                message_queue.put(f"Fetched metadata for ISBN {isbn}.")
            return {
                'author': authors if authors else None,
                'genre': genres if genres else None,
                'summary': summary if summary else None
            }
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching metadata for ISBN {isbn}: {e}")
        if message_queue:
            message_queue.put(f"Error fetching metadata for ISBN {isbn}: {e}")
    return {}


def impute_missing_metadata(df, message_queue=None, max_workers=10):
    """
    Imputes missing 'author' and 'genre' fields by fetching data from the OpenLibrary API in parallel.

    Args:
        df (pd.DataFrame): The DataFrame containing book data.
        message_queue (queue.Queue, optional): Queue for thread-safe GUI updates.
        max_workers (int, optional): Maximum number of threads for parallel processing.

    Returns:
        pd.DataFrame: The DataFrame with imputed metadata.
    """
    # Identify rows with missing 'author' or 'genre'
    missing_mask = df['author'].isna() | df['genre'].isna()
    missing_df = df[missing_mask].copy()

    def fetch_and_update(row):
        isbn = row['ISBN']
        metadata = fetch_book_metadata(isbn, message_queue)
        updates = {}
        if pd.isna(row['author']) and 'author' in metadata:
            updates['author'] = metadata['author']
        if pd.isna(row['genre']) and 'genre' in metadata:
            updates['genre'] = metadata['genre']
        return (row.name, updates)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(fetch_and_update, row) for _, row in missing_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            index, updates = future.result()
            if updates:
                if 'author' in updates:
                    df.at[index, 'author'] = updates['author']
                    logging.info(f"Imputed author for ISBN {df.at[index, 'ISBN']}.")
                    if message_queue:
                        message_queue.put(f"Imputed author for ISBN {df.at[index, 'ISBN']}.")
                if 'genre' in updates:
                    df.at[index, 'genre'] = updates['genre']
                    logging.info(f"Imputed genre for ISBN {df.at[index, 'ISBN']}.")
                    if message_queue:
                        message_queue.put(f"Imputed genre for ISBN {df.at[index, 'ISBN']}.")

    return df


def detect_and_remove_outliers(df, column='borrow_count', message_queue=None):
    if column not in df.columns or df[column].empty:
        message = f"No '{column}' column found or it is empty. Skipping outlier detection."
        logging.warning(message)
        if message_queue:
            message_queue.put(message)
        return message, df
    try:
        model = IForest()
        model.fit(df[[column]])
        outliers = model.predict(df[[column]])
        initial_count = df.shape[0]
        df = df[outliers == 1]
        final_count = df.shape[0]
        message = f"Removed {initial_count - final_count} outliers from '{column}'."
        logging.info(message)
        if message_queue:
            message_queue.put(message)
        return message, df
    except Exception as e:
        message = f"Outlier detection failed for '{column}': {e}"
        logging.error(message)
        if message_queue:
            message_queue.put(message)
        return message, df


def validate_data(df, message_queue=None):
    # Validate publication_year
    current_year = datetime.now().year
    df = df[df['publication_year'].between(1450, current_year)]
    message = f"Validated publication years between 1450 and {current_year}."
    logging.info(message)
    if message_queue:
        message_queue.put(message)

    # Enforce unique book_ids if applicable
    if 'book_id' in df.columns:
        initial_count = df.shape[0]
        df = df.drop_duplicates(subset=['book_id'], keep='first')
        final_count = df.shape[0]
        message = f"Removed {initial_count - final_count} duplicate book_ids."
        logging.info(message)
    else:
        message = "No 'book_id' column found. Skipping duplicate book_id enforcement."
        logging.info(message)

    if message_queue:
        message_queue.put(message)

    return message, df


def handle_missing_values(df, message_queue=None):
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        try:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            message = "Handled missing numeric values with KNN Imputation."
            logging.info(message)
            if message_queue:
                message_queue.put(message)
        except Exception as e:
            message = f"KNN Imputation failed: {e}"
            logging.error(message)
            if message_queue:
                message_queue.put(message)
    return df


def clean_library_data(input_file='library_books.csv', output_file='library_books_cleaned.csv', message_queue=None):
    # Load raw data
    if not os.path.exists(input_file):
        logging.error(f"Input file '{input_file}' not found.")
        if message_queue:
            message_queue.put(f"Input file '{input_file}' not found.")
        return None, None

    # Specify data types to ensure ISBN is read as a string
    try:
        df = pd.read_csv(input_file, dtype={'ISBN': str})
        logging.info(f"Loaded '{input_file}' with {df.shape[0]} records.")
        if message_queue:
            message_queue.put(f"Loaded '{input_file}' with {df.shape[0]} records.")
    except Exception as e:
        logging.error(f"Failed to read '{input_file}': {e}")
        if message_queue:
            message_queue.put(f"Failed to read '{input_file}': {e}")
        return None, None

    # Clean ISBNs
    df['ISBN'] = df['ISBN'].str.strip().str.replace('-', '').str.replace(' ', '')
    df = df.dropna(subset=['ISBN'])
    df['ISBN'] = df['ISBN'].astype(str)
    logging.info("Cleaned ISBNs by removing hyphens and spaces.")
    if message_queue:
        message_queue.put("Cleaned ISBNs by removing hyphens and spaces.")

    # Remove duplicates
    dup_msg, df = remove_duplicates(df, subset=['ISBN'])
    if message_queue:
        message_queue.put(dup_msg)

    # Remove records with critical missing data
    initial_count = df.shape[0]
    df = df.dropna(subset=['author', 'genre', 'publication_year'])
    final_count = df.shape[0]
    logging.info(f"Removed {initial_count - final_count} records with missing critical data.")
    if message_queue:
        message_queue.put(f"Removed {initial_count - final_count} records with missing critical data.")

    # Handle missing metadata
    df = impute_missing_metadata(df, message_queue)

    # Detect and remove outliers using PyOD
    outlier_msg, df = detect_and_remove_outliers(df, column='borrow_count', message_queue=message_queue)

    # Data validation
    validation_msg, df = validate_data(df, message_queue=message_queue)

    # Handle remaining missing values with KNN Imputation
    df = handle_missing_values(df, message_queue=message_queue)

    # Final cleaned data
    logging.info(f"Final cleaned records: {df.shape[0]}")
    if message_queue:
        message_queue.put(f"Final cleaned records: {df.shape[0]}")

    # Save cleaned data
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Cleaned data saved to '{output_file}'.")
        if message_queue:
            message_queue.put(f"Cleaned data saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to save cleaned data: {e}")
        if message_queue:
            message_queue.put(f"Failed to save cleaned data: {e}")
        return df, None

    return df, output_file

# ------------------------------
# Data Preparation and Feature Engineering Functions
# ------------------------------

def calculate_book_popularity(session, message_queue=None):
    """
    Calculates and updates the borrow count for all books in bulk.

    Args:
        session (Session): SQLAlchemy session object.
        message_queue (queue.Queue, optional): Queue for thread-safe GUI updates.
    """
    try:
        # Perform a single query to get borrow counts for all books
        borrow_counts = session.query(
            BorrowedBook.book_id,
            func.count(BorrowedBook.borrow_id).label('borrow_count')
        ).group_by(BorrowedBook.book_id).all()

        # Create a dictionary of book_id to borrow_count
        borrow_count_dict = {book_id: count for book_id, count in borrow_counts}

        # Update the borrow_count for all books
        session.query(Book).update({
            Book.borrow_count: func.coalesce(borrow_count_dict.get(Book.book_id, 0), 0)
        }, synchronize_session=False)

        session.commit()

        message = "Book popularity calculated and updated."
        logging.info(message)
        if message_queue:
            message_queue.put(message)
    except Exception as e:
        session.rollback()
        message = f"Failed to calculate book popularity: {e}"
        logging.error(message)
        if message_queue:
            message_queue.put(message)


def create_user_profiles(session, message_queue=None):
    try:
        users = session.query(User).all()
        if not users:
            message = "No users found in the database to create profiles."
            logging.warning(message)
            if message_queue:
                message_queue.put(message)
            return {}
        user_profiles = {}
        for user in users:
            borrowed_books = session.query(BorrowedBook).filter_by(user_id=user.user_id).all()
            genres = []
            for borrow in borrowed_books:
                book = session.query(Book).filter_by(book_id=borrow.book_id).first()
                if book:
                    genres.append(book.genre)
            if genres:
                genre_counts = pd.Series(genres).value_counts().to_dict()
                user_profiles[user.user_id] = genre_counts
                logging.debug(f"Created profile for user_id {user.user_id}.")
                if message_queue:
                    message_queue.put(f"Created profile for user_id {user.user_id}.")
            else:
                user_profiles[user.user_id] = {}
                logging.debug(f"No borrowed books for user_id {user.user_id}.")
                if message_queue:
                    message_queue.put(f"No borrowed books for user_id {user.user_id}.")
        return user_profiles
    except Exception as e:
        message = f"Failed to create user profiles: {e}"
        logging.error(message)
        if message_queue:
            message_queue.put(message)
        return {}


def save_user_profiles(user_profiles, output_file='user_profiles.csv', message_queue=None):
    try:
        profiles = []
        for user_id, genres in user_profiles.items():
            profile = {'user_id': user_id}
            for genre, count in genres.items():
                profile[genre] = count
            profiles.append(profile)
        profiles_df = pd.DataFrame(profiles).fillna(0)
        profiles_df.to_csv(output_file, index=False)
        message = f"User profiles saved to '{output_file}'."
        logging.info(message)
        if message_queue:
            message_queue.put(message)
    except Exception as e:
        message = f"Failed to save user profiles: {e}"
        logging.error(message)
        if message_queue:
            message_queue.put(message)


def feature_engineering(session, message_queue=None):
    calculate_book_popularity(session, message_queue)
    user_profiles = create_user_profiles(session, message_queue)
    if user_profiles:
        save_user_profiles(user_profiles, message_queue=message_queue)
    logging.info("Feature engineering completed.")
    if message_queue:
        message_queue.put("Feature engineering completed.")

# ------------------------------
# Recommendation Engine Module
# ------------------------------

class ContentBasedRecommender:
    def __init__(self, books_df):
        self.books_df = books_df.reset_index(drop=True)  # Ensure index starts at 0
        self.books_df['features'] = (
                self.books_df['genre'] + ' ' +
                self.books_df['author'] + ' ' +
                self.books_df['keywords'].fillna('')
        )
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df['features'])
        self.cos_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.books_df.index, index=self.books_df['title']).drop_duplicates()
        logging.info("Content-Based Recommender initialized.")

    def get_recommendations(self, book_title, top_n=5):
        if book_title not in self.indices:
            logging.warning(f"'{book_title}' not found in the database.")
            return []
        idx = self.indices[book_title]
        sim_scores = list(enumerate(self.cos_sim[idx].flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Exclude the book itself
        book_indices = [i[0] for i in sim_scores if i[0] < len(self.books_df)]
        if not book_indices:
            return []
        recommendations = self.books_df['title'].iloc[book_indices].tolist()
        logging.info(f"Content-Based Recommendations for '{book_title}': {recommendations}")
        return recommendations


class CollaborativeFilteringRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        if self.ratings_df.empty or len(self.ratings_df['user_id'].unique()) < 2 or len(
                self.ratings_df['book_id'].unique()) < 2:
            self.algo = None
            self.trainset = None
            self.testset = None
            self.predictions = []
            self.rmse = None
            logging.warning("Insufficient ratings data to train the collaborative filtering model.")
        else:
            try:
                reader = Reader(rating_scale=(1, 5))
                self.data = Dataset.load_from_df(self.ratings_df[['user_id', 'book_id', 'rating']], reader)
                self.trainset, self.testset = train_test_split(self.data, test_size=0.2)
                self.algo = SVD()
                self.algo.fit(self.trainset)
                self.predictions = self.algo.test(self.testset)
                self.rmse = accuracy.rmse(self.predictions, verbose=False)
                logging.info(f"Collaborative Filtering Model trained with RMSE: {self.rmse:.4f}")
            except Exception as e:
                logging.error(f"Failed to train Collaborative Filtering model: {e}")
                self.algo = None

    def get_recommendations(self, user_id, books_df, top_n=5):
        if self.algo is None:
            logging.warning("Collaborative Filtering model is not trained.")
            return []

        try:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].tolist()
            all_books = books_df['book_id'].tolist()
            books_to_predict = list(set(all_books) - set(user_ratings))
            if not books_to_predict:
                logging.info(f"No new books to recommend for user_id {user_id}.")
                return []
            predictions = [self.algo.predict(user_id, book_id) for book_id in books_to_predict]
            predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
            top_predictions = predictions[:top_n]
            recommended_book_ids = [pred.iid for pred in top_predictions]
            recommended_books = books_df[books_df['book_id'].isin(recommended_book_ids)]['title'].tolist()
            logging.info(f"Collaborative Filtering Recommendations for user_id {user_id}: {recommended_books}")
            return recommended_books
        except Exception as e:
            logging.error(f"Failed to get Collaborative Filtering recommendations: {e}")
            return []

    def evaluate_model(self):
        if self.algo is None or not self.predictions:
            logging.warning("Collaborative Filtering model is not trained or has no predictions.")
            return None
        try:
            rmse = accuracy.rmse(self.predictions, verbose=False)
            logging.info(f"Collaborative Filtering Model RMSE after evaluation: {rmse:.4f}")
            return rmse
        except Exception as e:
            logging.error(f"Failed to evaluate Collaborative Filtering model: {e}")
            return None


class HybridRecommender:
    def __init__(self, content_recommender, cf_recommender, books_df):
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender
        self.books_df = books_df
        logging.info("Hybrid Recommender initialized.")

    def get_recommendations(self, user_id, book_title, top_n=5, weight_content=0.5, weight_cf=0.5):
        content_recs = self.content_recommender.get_recommendations(book_title, top_n=top_n * 2)
        cf_recs = self.cf_recommender.get_recommendations(user_id, self.books_df, top_n=top_n * 2)

        # Combine recommendations with weights
        recommendation_scores = {}

        for rec in content_recs:
            recommendation_scores[rec] = recommendation_scores.get(rec, 0) + weight_content

        for rec in cf_recs:
            recommendation_scores[rec] = recommendation_scores.get(rec, 0) + weight_cf

        # Sort based on combined scores
        sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = [rec[0] for rec in sorted_recs[:top_n]]

        logging.info(f"Hybrid Recommendations for user_id {user_id} based on '{book_title}': {top_recommendations}")
        return top_recommendations

    def recommend_for_new_user(self, books_df, top_n=5):
        # For new users, recommend the most popular books
        popular_books = books_df.sort_values(by='borrow_count', ascending=False).head(top_n)
        recommendations = popular_books['title'].tolist()
        logging.info(f"Popular Recommendations for new users: {recommendations}")
        return recommendations

# ------------------------------
# Recommendation System Initialization
# ------------------------------

def initialize_recommender_system(message_queue=None):
    # Step 1: Clean Data
    df, cleaned_data_file = clean_library_data(message_queue=message_queue)

    if not cleaned_data_file:
        logging.error("Data cleaning failed. Cannot proceed with initialization.")
        if message_queue:
            message_queue.put("Data cleaning failed. Cannot proceed with initialization.")
        return None, None, None, None

    # Step 2: Setup Database
    session = setup_database()
    if session is None:
        logging.error("Database setup failed. Cannot proceed with initialization.")
        if message_queue:
            message_queue.put("Database setup failed. Cannot proceed with initialization.")
        return None, None, None, None

    # Step 3: Populate Database
    populate_database(session, cleaned_books_file=cleaned_data_file, message_queue=message_queue)

    # Step 4: Feature Engineering
    feature_engineering(session, message_queue=message_queue)

    # Step 5: Load Data for Recommendation Engine
    try:
        books_df = pd.read_sql(session.query(Book).statement, session.bind)
        ratings_df = pd.read_sql(session.query(Rating).statement, session.bind)
        logging.info("Loaded books and ratings data for recommendation engine.")
        if message_queue:
            message_queue.put("Loaded books and ratings data for recommendation engine.")
    except Exception as e:
        logging.error(f"Failed to load data for recommendation engine: {e}")
        if message_queue:
            message_queue.put(f"Failed to load data for recommendation engine: {e}")
        session.close()
        return None, None, None, None

    session.close()

    # Step 6: Initialize Recommenders
    content_recommender = ContentBasedRecommender(books_df)
    cf_recommender = CollaborativeFilteringRecommender(ratings_df)
    hybrid_recommender = HybridRecommender(content_recommender, cf_recommender, books_df)

    # Check if CF model is trained
    if cf_recommender.algo is None:
        logging.warning("Collaborative Filtering model was not trained due to insufficient data.")
        if message_queue:
            message_queue.put("Collaborative Filtering model was not trained due to insufficient data.")

    logging.info("Recommendation system initialized successfully.")
    if message_queue:
        message_queue.put("Recommendation system initialized successfully.")
    return hybrid_recommender, books_df, ratings_df, cf_recommender

# ------------------------------
# Tkinter GUI Classes with Enhanced Aesthetic
# ------------------------------
def populate_database(session, cleaned_books_file, message_queue=None):
    """
    Populates the 'books' table in the database with data from the cleaned_books_file.

    Args:
        session (Session): SQLAlchemy session object.
        cleaned_books_file (str): Path to the cleaned books CSV file.
        message_queue (queue.Queue, optional): Queue for thread-safe GUI updates.
    """
    try:
        # Load the cleaned data
        df = pd.read_csv(cleaned_books_file)
        logging.info(f"Loaded '{cleaned_books_file}' with {df.shape[0]} records for database population.")
        if message_queue:
            message_queue.put(f"Loaded '{cleaned_books_file}' with {df.shape[0]} records for database population.")

        # Get existing ISBNs from the database
        existing_isbns = set(book.ISBN for book in session.query(Book.ISBN).all())
        logging.info(f"Found {len(existing_isbns)} existing ISBNs in the database.")
        if message_queue:
            message_queue.put(f"Found {len(existing_isbns)} existing ISBNs in the database.")

        # Filter out books with existing ISBNs
        new_books_df = df[~df['ISBN'].astype(str).isin(existing_isbns)]
        num_new_books = new_books_df.shape[0]
        logging.info(f"{num_new_books} new books to insert into the database.")
        if message_queue:
            message_queue.put(f"{num_new_books} new books to insert into the database.")

        # Prepare Book objects
        books = []
        for _, row in new_books_df.iterrows():
            book = Book(
                title=row['title'],
                author=row['author'],
                genre=row['genre'],
                ISBN=row['ISBN'],
                publication_year=int(row['publication_year']),
                summary=row.get('summary', ''),
                keywords=row.get('keywords', ''),
                borrow_count=int(row.get('borrow_count', 0))
            )
            books.append(book)

        if books:
            # Bulk save for efficiency
            session.bulk_save_objects(books)
            session.commit()

            message = f"Populated database with {len(books)} new books."
            logging.info(message)
            if message_queue:
                message_queue.put(message)
        else:
            message = "No new books to populate."
            logging.info(message)
            if message_queue:
                message_queue.put(message)

    except Exception as e:
        session.rollback()
        message = f"Failed to populate database: {e}"
        logging.error(message)
        if message_queue:
            # Capture the exception message correctly
            error_message = f"Failed to populate database: {e}"
            self.queue.put(lambda error_message=error_message: messagebox.showerror("Error", error_message))
        raise e  # Re-raise exception to handle it in the calling function

class LoginWindow:
    def __init__(self, master, app):
        self.master = master
        self.app = app  # Reference to the main application
        self.master.title("Library Recommender System - Login")
        self.master.geometry("500x600")
        self.master.configure(bg="#f0f8ff")  # AliceBlue background

        # Load and place logo
        self.logo_photo = load_image(LOGO_PATH, (150, 150))
        if self.logo_photo:
            tk.Label(self.master, image=self.logo_photo, bg="#f0f8ff").pack(pady=20)
        else:
            # Use placeholder text if image fails to load
            tk.Label(self.master, text="📚 Library Logo", font=('Helvetica', 16, 'bold'), bg="#f0f8ff").pack(pady=20)

        # Create a frame for the login form
        login_frame = ttk.Frame(self.master, padding="30 30 30 30")
        login_frame.pack(fill=tk.BOTH, expand=True)

        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', background="#f0f8ff")
        style.configure('TLabel', font=('Helvetica', 12), background="#f0f8ff")
        style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground="#000", background="#fff")
        style.map('TButton',
                  foreground=[('active', '#000')],
                  background=[('active', '#ccc')])

        # Email
        ttk.Label(login_frame, text="Email:").grid(row=0, column=0, sticky=tk.W, pady=15)
        self.email_entry = ttk.Entry(login_frame, font=('Helvetica', 12), width=30)
        self.email_entry.grid(row=0, column=1, pady=15, padx=10)
        self.email_entry.focus()

        # Password
        ttk.Label(login_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, pady=15)
        self.password_entry = ttk.Entry(login_frame, show="*", font=('Helvetica', 12), width=30)
        self.password_entry.grid(row=1, column=1, pady=15, padx=10)

        # Login Button with Icon and Animation
        self.login_photo = load_image(LOGIN_ICON_PATH, (20, 20))
        if self.login_photo:
            self.login_button = ttk.Button(login_frame, text=" Login", image=self.login_photo, compound='left',
                                           command=self.login)
        else:
            self.login_button = ttk.Button(login_frame, text="Login", command=self.login)

        self.login_button.grid(row=2, column=0, columnspan=2, pady=30, ipadx=10)
        self.add_button_hover_effect(self.login_button)

        # Register Button with Icon and Animation
        self.register_photo = load_image(REGISTER_ICON_PATH, (20, 20))
        if self.register_photo:
            self.register_button = ttk.Button(register_frame := ttk.Frame(self.master, padding="30 30 30 30"),
                                              text=" Register", image=self.register_photo, compound='left',
                                              command=self.open_register_window)
        else:
            self.register_button = ttk.Button(register_frame := ttk.Frame(self.master, padding="30 30 30 30"),
                                              text="Register", command=self.open_register_window)

        self.register_button.grid(row=3, column=0, columnspan=2, pady=10, ipadx=10)
        self.add_button_hover_effect(self.register_button)

        # Apply fade-in animation
        self.fade_in_animation()

    def fade_in_animation(self, delay=0, steps=10):
        def fade(step=0):
            if step <= steps:
                opacity = step / steps
                self.master.attributes("-alpha", opacity)
                self.master.after(delay, lambda: fade(step + 1))
            else:
                self.master.attributes("-alpha", 1.0)

        # Ensure the window has the '-alpha' attribute
        try:
            self.master.attributes("-alpha", 0.0)
            fade()
        except Exception as e:
            logging.warning(f"Fade-in animation failed: {e}")

    def add_button_hover_effect(self, button):
        button.bind("<Enter>", lambda e: self.animate_button(button, hover=True))
        button.bind("<Leave>", lambda e: self.animate_button(button, hover=False))

    def animate_button(self, button, hover=True):
        if hover:
            button.config(style='Hovered.TButton')
        else:
            button.config(style='TButton')

    def login(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()

        if not email or not password:
            messagebox.showerror("Error", "Please enter both email and password.")
            return

        # Connect to database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        user = session.query(User).filter_by(email=email).first()
        session.close()

        if user and verify_password(password, user.password):
            messagebox.showinfo("Success", f"Welcome back, {user.name}!")
            # Proceed to main application
            self.app.open_main_app(user)
        else:
            messagebox.showerror("Error", "Invalid email or password.")

    def open_register_window(self):
        self.app.open_register_window()


class RegisterWindow:
    def __init__(self, master, app):
        self.master = master
        self.app = app  # Reference to the main application
        self.master.title("Library Recommender System - Register")
        self.master.geometry("500x700")
        self.master.configure(bg="#f0f8ff")  # AliceBlue background

        # Load and place logo
        self.logo_photo = load_image(LOGO_PATH, (150, 150))
        if self.logo_photo:
            tk.Label(self.master, image=self.logo_photo, bg="#f0f8ff").pack(pady=20)
        else:
            # Use placeholder text if image fails to load
            tk.Label(self.master, text="📚 Library Logo", font=('Helvetica', 16, 'bold'), bg="#f0f8ff").pack(pady=20)

        # Create a frame for the registration form
        register_frame = ttk.Frame(self.master, padding="30 30 30 30")
        register_frame.pack(fill=tk.BOTH, expand=True)

        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', background="#f0f8ff")
        style.configure('TLabel', font=('Helvetica', 12), background="#f0f8ff")
        style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground="#000", background="#fff")
        style.map('TButton',
                  foreground=[('active', '#000')],
                  background=[('active', '#ccc')])
        style.configure('Hovered.TButton', background="#333", foreground="#fff")

        # Name
        ttk.Label(register_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=15)
        self.name_entry = ttk.Entry(register_frame, font=('Helvetica', 12), width=30)
        self.name_entry.grid(row=0, column=1, pady=15, padx=10)

        # Email
        ttk.Label(register_frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=15)
        self.email_entry = ttk.Entry(register_frame, font=('Helvetica', 12), width=30)
        self.email_entry.grid(row=1, column=1, pady=15, padx=10)

        # Password
        ttk.Label(register_frame, text="Password:").grid(row=2, column=0, sticky=tk.W, pady=15)
        self.password_entry = ttk.Entry(register_frame, show="*", font=('Helvetica', 12), width=30)
        self.password_entry.grid(row=2, column=1, pady=15, padx=10)

        # Confirm Password
        ttk.Label(register_frame, text="Confirm Password:").grid(row=3, column=0, sticky=tk.W, pady=15)
        self.confirm_password_entry = ttk.Entry(register_frame, show="*", font=('Helvetica', 12), width=30)
        self.confirm_password_entry.grid(row=3, column=1, pady=15, padx=10)

        # Register Button with Icon and Animation
        self.register_photo = load_image(REGISTER_ICON_PATH, (20, 20))
        if self.register_photo:
            self.register_button = ttk.Button(register_frame, text=" Register", image=self.register_photo,
                                              compound='left', command=self.register, style='TButton')
        else:
            self.register_button = ttk.Button(register_frame, text="Register", command=self.register, style='TButton')

        self.register_button.grid(row=4, column=0, columnspan=2, pady=40, ipadx=10)
        self.add_button_hover_effect(self.register_button)

        # Back to Login Button with Icon and Animation
        self.back_photo = load_image(LOGIN_ICON_PATH, (20, 20))
        if self.back_photo:
            self.back_button = ttk.Button(register_frame, text=" Back to Login", image=self.back_photo, compound='left',
                                          command=self.back_to_login, style='TButton')
        else:
            self.back_button = ttk.Button(register_frame, text="Back to Login", command=self.back_to_login,
                                          style='TButton')

        self.back_button.grid(row=5, column=0, columnspan=2, pady=10, ipadx=10)
        self.add_button_hover_effect(self.back_button)

        # Apply fade-in animation
        self.fade_in_animation()

    def fade_in_animation(self, delay=0, steps=10):
        def fade(step=0):
            if step <= steps:
                opacity = step / steps
                self.master.attributes("-alpha", opacity)
                self.master.after(delay, lambda: fade(step + 1))
            else:
                self.master.attributes("-alpha", 1.0)

        # Ensure the window has the '-alpha' attribute
        try:
            self.master.attributes("-alpha", 0.0)
            fade()
        except Exception as e:
            logging.warning(f"Fade-in animation failed: {e}")

    def add_button_hover_effect(self, button):
        button.bind("<Enter>", lambda e: self.animate_button(button, hover=True))
        button.bind("<Leave>", lambda e: self.animate_button(button, hover=False))

    def animate_button(self, button, hover=True):
        if hover:
            button.config(style='Hovered.TButton')
        else:
            button.config(style='TButton')

    def register(self):
        name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()
        confirm_password = self.confirm_password_entry.get().strip()

        if not name or not email or not password or not confirm_password:
            messagebox.showerror("Error", "All fields are required.")
            return

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match.")
            return

        # Hash the password
        hashed_password = hash_password(password).decode('utf-8')

        # Connect to database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        # Check if email already exists
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            messagebox.showerror("Error", "Email already registered.")
            session.close()
            return

        # Create new user
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            registration_date=datetime.now().date()
        )
        try:
            session.add(new_user)
            session.commit()
            logging.info(f"New user registered: {email}")
            messagebox.showinfo("Success", "Registration successful! You can now log in.")
            # Close the registration window
            self.master.destroy()
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to register user: {e}")
            messagebox.showerror("Error", f"Registration failed: {e}")
        finally:
            session.close()

    def back_to_login(self):
        # Close the registration window
        self.master.destroy()


class MainApp:
    def __init__(self, master, user):
        self.master = master
        self.user = user
        self.master.title("Library Recommender System")
        self.master.geometry("1200x800")
        self.master.configure(bg="#f0f8ff")  # AliceBlue background
        self.recommender_system = None

        # Initialize the queue for thread-safe GUI updates
        self.queue = queue.Queue()
        self.master.after(100, self.process_queue)

        # Configure styles for hovered buttons
        style = ttk.Style()
        style.configure('Hovered.TButton', background="#333", foreground="#fff")
        style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground="#000", background="#fff")
        style.map('Hovered.TButton',
                  background=[('active', '#555')],
                  foreground=[('active', '#fff')])

        # Initialize status_var first
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to the Library Recommender System!")
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                               background="#f0f8ff", font=('Helvetica', 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Load and place header image
        self.header_photo = load_image(HEADER_ICON_PATH, (1200, 150))
        if self.header_photo:
            tk.Label(self.master, image=self.header_photo, bg="#f0f8ff").pack(pady=10)
        else:
            # Use placeholder text if image fails to load
            tk.Label(self.master, text="📖 Library Header", font=('Helvetica', 16, 'bold'), bg="#f0f8ff").pack(pady=10)

        # Create Notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=10, expand=True)

        # Create frames for each tab
        self.init_tab = ttk.Frame(self.notebook, padding="20 20 20 20")
        self.recommend_tab = ttk.Frame(self.notebook, padding="20 20 20 20")
        self.new_user_tab = ttk.Frame(self.notebook, padding="20 20 20 20")
        self.submit_rating_tab = ttk.Frame(self.notebook, padding="20 20 20 20")
        self.evaluate_tab = ttk.Frame(self.notebook, padding="20 20 20 20")
        self.profile_tab = ttk.Frame(self.notebook, padding="20 20 20 20")  # New Profile Tab
        self.my_ratings_tab = ttk.Frame(self.notebook, padding="20 20 20 20")  # New My Ratings Tab
        self.charts_tab = ttk.Frame(self.notebook, padding="20 20 20 20")  # New Charts Tab

        # Add tabs to the notebook
        self.notebook.add(self.init_tab, text='Initialize System')
        self.notebook.add(self.recommend_tab, text='Get Recommendations')
        self.notebook.add(self.new_user_tab, text='New User Recommendations')
        self.notebook.add(self.submit_rating_tab, text='Submit Rating')
        self.notebook.add(self.evaluate_tab, text='Evaluate Model')
        self.notebook.add(self.profile_tab, text='User Profile')  # Adding Profile Tab
        self.notebook.add(self.my_ratings_tab, text='My Ratings')  # Adding My Ratings Tab
        self.notebook.add(self.charts_tab, text='Charts')  # Adding Charts Tab

        # Welcome Label
        ttk.Label(self.master, text=f"Welcome, {self.user.name}!", font=('Helvetica', 16, 'bold'),
                  background="#f0f8ff").pack(pady=10)

        # Add Logout Button
        self.logout_button = ttk.Button(self.master, text="Logout", command=self.logout)
        self.logout_button.pack(pady=5, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.logout_button)

        # Initialize all tabs
        self.create_init_tab()
        self.create_recommend_tab()
        self.create_new_user_tab()
        self.create_submit_rating_tab()
        self.create_evaluate_tab()
        self.create_profile_tab()
        self.create_my_ratings_tab()
        self.create_charts_tab()

    def add_button_hover_effect(self, button):
        button.bind("<Enter>", lambda e: self.animate_button(button, hover=True))
        button.bind("<Leave>", lambda e: self.animate_button(button, hover=False))

    def animate_button(self, button, hover=True):
        if hover:
            button.config(style='Hovered.TButton')
        else:
            button.config(style='TButton')

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if callable(item):
                    # If the item is a callable, execute it
                    item()
                elif isinstance(item, str):
                    # If the item is a string, append it to the log
                    self.init_log_text.configure(state='normal')
                    self.init_log_text.insert(tk.END, item + '\n')
                    self.init_log_text.configure(state='disabled')
                else:
                    logging.warning(f"Unhandled item in queue: {item}")
        except queue.Empty:
            pass
        # Continue checking the queue
        self.master.after(100, self.process_queue)

    def update_status(self, message):
        self.status_var.set(message)

    def create_init_tab(self):
        # Initialize System Button
        self.init_button = ttk.Button(self.init_tab, text="Initialize System", command=self.initialize_system)
        self.init_button.pack(pady=10, ipadx=20, ipady=10)

        # Description Label
        ttk.Label(
            self.init_tab,
            text="Click the button above to initialize the recommendation system.\nThis may take a while depending on the dataset size.",
            justify=tk.CENTER,
            font=('Helvetica', 12)
        ).pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.init_tab, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=10)

        # Log Text Widget to display cleaning progress
        ttk.Label(self.init_tab, text="Initialization Log:", font=('Helvetica', 12, 'bold')).pack(pady=10)
        self.init_log_text = tk.Text(self.init_tab, height=15, width=100, bg="#e6f0ff", font=('Helvetica', 10),
                                     wrap=tk.WORD)
        self.init_log_text.pack(pady=5)
        self.init_log_text.configure(state='disabled')  # Make it read-only

    def create_recommend_tab(self):
        # Book Title
        ttk.Label(self.recommend_tab, text="Enter Book Title:").grid(row=0, column=0, sticky=tk.W, pady=15, padx=20)
        self.book_title_entry = ttk.Entry(self.recommend_tab, font=('Helvetica', 12), width=40)
        self.book_title_entry.grid(row=0, column=1, pady=15, padx=10)
        self.book_title_entry.focus()

        # Number of Recommendations
        ttk.Label(self.recommend_tab, text="Number of Recommendations:").grid(row=1, column=0, sticky=tk.W, pady=15,
                                                                              padx=20)
        self.recommend_top_n = tk.IntVar(value=5)
        self.recommend_spinbox = ttk.Spinbox(self.recommend_tab, from_=1, to=20, textvariable=self.recommend_top_n,
                                             width=5)
        self.recommend_spinbox.grid(row=1, column=1, sticky=tk.W, pady=15, padx=10)

        # Get Recommendations Button
        self.get_recommend_button = ttk.Button(self.recommend_tab, text="Get Recommendations",
                                               command=self.get_recommendations)
        self.get_recommend_button.grid(row=2, column=0, columnspan=2, pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.get_recommend_button)

        # Recommendations Display
        self.recommend_text = tk.Text(self.recommend_tab, height=25, width=100, bg="#e6f0ff", font=('Helvetica', 12),
                                      wrap=tk.WORD)
        self.recommend_text.grid(row=3, column=0, columnspan=2, padx=20, pady=10)
        self.recommend_text.configure(state='disabled')  # Make it read-only

    def create_new_user_tab(self):
        # Number of Recommendations
        ttk.Label(self.new_user_tab, text="Number of Recommendations:").pack(pady=20, padx=20, anchor=tk.W)
        self.new_user_top_n = tk.IntVar(value=5)
        self.new_user_spinbox = ttk.Spinbox(self.new_user_tab, from_=1, to=20, textvariable=self.new_user_top_n,
                                            width=5)
        self.new_user_spinbox.pack(pady=10, padx=20, anchor=tk.W)

        # Get Recommendations Button
        self.get_new_user_recommend_button = ttk.Button(self.new_user_tab, text="Get Recommendations",
                                                        command=self.get_new_user_recommendations)
        self.get_new_user_recommend_button.pack(pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.get_new_user_recommend_button)

        # Recommendations Display
        self.new_user_recs_text = tk.Text(self.new_user_tab, height=25, width=100, bg="#e6f0ff",
                                          font=('Helvetica', 12),
                                          wrap=tk.WORD)
        self.new_user_recs_text.pack(padx=20, pady=10)
        self.new_user_recs_text.configure(state='disabled')  # Make it read-only

    def create_submit_rating_tab(self):
        # Book Title
        ttk.Label(self.submit_rating_tab, text="Enter Book Title:").grid(row=0, column=0, sticky=tk.W, pady=20, padx=20)
        self.submit_book_title_entry = ttk.Entry(self.submit_rating_tab, font=('Helvetica', 12), width=40)
        self.submit_book_title_entry.grid(row=0, column=1, pady=20, padx=20)

        # Rating
        ttk.Label(self.submit_rating_tab, text="Rating (1-5):").grid(row=1, column=0, sticky=tk.W, pady=20, padx=20)
        self.submit_rating_var = tk.DoubleVar(value=3.0)
        self.submit_rating_spinbox = ttk.Spinbox(self.submit_rating_tab, from_=1.0, to=5.0, increment=0.5,
                                                 textvariable=self.submit_rating_var, width=5)
        self.submit_rating_spinbox.grid(row=1, column=1, sticky=tk.W, pady=20, padx=20)

        # Submit Rating Button
        self.submit_rating_button = ttk.Button(self.submit_rating_tab, text="Submit Rating", command=self.submit_rating)
        self.submit_rating_button.grid(row=2, column=0, columnspan=2, pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.submit_rating_button)

    def create_evaluate_tab(self):
        # Run Evaluation Button
        self.run_evaluation_button = ttk.Button(self.evaluate_tab, text="Run Evaluation", command=self.evaluate_model)
        self.run_evaluation_button.pack(pady=10, ipadx=20, ipady=10)
        self.add_button_hover_effect(self.run_evaluation_button)

        # Description Label
        ttk.Label(
            self.evaluate_tab,
            text="Click the button above to evaluate the collaborative filtering model using RMSE.\nEnsure that ratings data is available.",
            justify=tk.CENTER,
            font=('Helvetica', 12)
        ).pack(pady=10)

        # Evaluation Result Display
        self.evaluation_result_text = tk.Text(self.evaluate_tab, height=10, width=100, bg="#e6f0ff",
                                              font=('Helvetica', 12),
                                              wrap=tk.WORD)
        self.evaluation_result_text.pack(padx=20, pady=10)
        self.evaluation_result_text.configure(state='disabled')  # Make it read-only

    def create_profile_tab(self):
        # Profile Information
        profile_frame = ttk.Frame(self.profile_tab, padding="20 20 20 20")
        profile_frame.pack(fill=tk.BOTH, expand=True)

        # Name
        ttk.Label(profile_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=15)
        self.profile_name_entry = ttk.Entry(profile_frame, font=('Helvetica', 12), width=30)
        self.profile_name_entry.grid(row=0, column=1, pady=15, padx=10)
        self.profile_name_entry.insert(0, self.user.name)

        # Email
        ttk.Label(profile_frame, text="Email:").grid(row=1, column=0, sticky=tk.W, pady=15)
        self.profile_email_entry = ttk.Entry(profile_frame, font=('Helvetica', 12), width=30)
        self.profile_email_entry.grid(row=1, column=1, pady=15, padx=10)
        self.profile_email_entry.insert(0, self.user.email)

        # Password
        ttk.Label(profile_frame, text="New Password:").grid(row=2, column=0, sticky=tk.W, pady=15)
        self.profile_password_entry = ttk.Entry(profile_frame, show="*", font=('Helvetica', 12), width=30)
        self.profile_password_entry.grid(row=2, column=1, pady=15, padx=10)

        # Confirm New Password
        ttk.Label(profile_frame, text="Confirm New Password:").grid(row=3, column=0, sticky=tk.W, pady=15)
        self.profile_confirm_password_entry = ttk.Entry(profile_frame, show="*", font=('Helvetica', 12), width=30)
        self.profile_confirm_password_entry.grid(row=3, column=1, pady=15, padx=10)

        # Update Profile Button with Animation
        self.update_profile_button = ttk.Button(profile_frame, text="Update Profile", command=self.update_profile,
                                                style='TButton')
        self.update_profile_button.grid(row=4, column=0, columnspan=2, pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.update_profile_button)

    def create_my_ratings_tab(self):
        # Fetch and display user's ratings
        self.my_ratings_tree = ttk.Treeview(self.my_ratings_tab, columns=("Book Title", "Rating", "Date"),
                                            show='headings')
        self.my_ratings_tree.heading("Book Title", text="Book Title")
        self.my_ratings_tree.heading("Rating", text="Rating")
        self.my_ratings_tree.heading("Date", text="Date")
        self.my_ratings_tree.column("Book Title", anchor=tk.W, width=500)
        self.my_ratings_tree.column("Rating", anchor=tk.CENTER, width=100)
        self.my_ratings_tree.column("Date", anchor=tk.CENTER, width=150)
        self.my_ratings_tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.my_ratings_tab, orient=tk.VERTICAL, command=self.my_ratings_tree.yview)
        self.my_ratings_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate the treeview
        self.populate_my_ratings()

    def create_charts_tab(self):
        # Chart for Book Popularity
        chart_frame = ttk.Frame(self.charts_tab, padding="20 20 20 20")
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # Label
        ttk.Label(chart_frame, text="📈 Book Popularity Chart", font=('Helvetica', 14, 'bold')).pack(pady=10)

        # Placeholder for the chart
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button to Generate Chart
        self.generate_chart_button = ttk.Button(chart_frame, text="Generate Chart", command=self.generate_chart)
        self.generate_chart_button.pack(pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.generate_chart_button)

    def initialize_system(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            # Disable the button to prevent multiple clicks
            self.init_button.config(state='disabled')
            self.update_status("Initializing the recommendation system. Please wait...")
            self.progress.start(10)  # Start the progress bar

            # Clear previous logs
            self.init_log_text.configure(state='normal')
            self.init_log_text.delete(1.0, tk.END)
            self.init_log_text.configure(state='disabled')

            # Run initialization in a separate daemon thread to keep the GUI responsive
            threading.Thread(target=self.run_initialization, daemon=True).start()
        else:
            messagebox.showwarning("Warning", "Recommendation system is already initialized.")

    def run_initialization(self):
        try:
            self.recommender_system, _, _, _ = initialize_recommender_system(message_queue=self.queue)
            # Put the callback in the queue
            self.queue.put(self.post_initialization)
        except Exception as e:
            # Handle exceptions and notify the main thread
            self.queue.put(lambda: messagebox.showerror("Error", f"Initialization failed: {e}"))

    def post_initialization(self):
        self.progress.stop()
        if self.recommender_system:
            self.update_status("Recommendation system initialized successfully.")
            messagebox.showinfo("Success", "Recommendation system initialized successfully.")

            # Enable Evaluate Model tab if CF model is trained
            # Tabs are 0-indexed: Evaluate Model is the 4th tab (after Initialize, Recommend, New User, Submit Rating)
            if self.recommender_system.cf_recommender.algo is not None:
                self.notebook.tab(4, state='normal')
            else:
                self.notebook.tab(4, state='disabled')
                messagebox.showwarning("Warning",
                                       "Collaborative Filtering model was not trained due to insufficient ratings data.")

            # Enable Charts tab
            self.notebook.tab(6, state='normal')
        else:
            self.update_status("Failed to initialize the recommendation system.")
            messagebox.showerror("Error", "Failed to initialize the recommendation system.")
        # Re-enable the button
        self.init_button.config(state='normal')

    def get_recommendations(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        book_title = self.book_title_entry.get().strip()
        top_n = self.recommend_top_n.get()

        if not book_title:
            messagebox.showerror("Error", "Book title cannot be empty.")
            return

        recommendations = self.recommender_system.get_recommendations(
            user_id=self.user.user_id,
            book_title=book_title,
            top_n=top_n
        )
        self.recommend_text.configure(state='normal')
        self.recommend_text.delete(1.0, tk.END)
        if recommendations:
            self.recommend_text.insert(tk.END, f"📚 Recommended Books based on '{book_title}':\n\n")
            for idx, book in enumerate(recommendations, 1):
                self.recommend_text.insert(tk.END, f"{idx}. {book}\n")
        else:
            self.recommend_text.insert(tk.END, "No recommendations found for the entered book title.")
            logging.info(f"'{book_title}' not found in the database or no similar books available.")
        self.recommend_text.configure(state='disabled')

    def get_new_user_recommendations(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        top_n = self.new_user_top_n.get()
        recommendations = self.recommender_system.recommend_for_new_user(
            books_df=self.recommender_system.books_df,
            top_n=top_n
        )
        self.new_user_recs_text.configure(state='normal')
        self.new_user_recs_text.delete(1.0, tk.END)
        if recommendations:
            self.new_user_recs_text.insert(tk.END, "📚 Recommended Books for New Users:\n\n")
            for idx, book in enumerate(recommendations, 1):
                self.new_user_recs_text.insert(tk.END, f"{idx}. {book}\n")
        else:
            self.new_user_recs_text.insert(tk.END, "No recommendations found.")
        self.new_user_recs_text.configure(state='disabled')

    def submit_rating(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        book_title = self.submit_book_title_entry.get().strip()
        user_rating = self.submit_rating_var.get()

        if not book_title:
            messagebox.showerror("Error", "Book title cannot be empty.")
            return

        # Fetch book_id from title
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        book = session.query(Book).filter_by(title=book_title).first()
        session.close()

        if not book:
            messagebox.showerror("Error", f"Book titled '{book_title}' not found.")
            return

        # Insert rating into database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        try:
            # Check if user has already rated this book
            existing_rating = session.query(Rating).filter_by(user_id=self.user.user_id, book_id=book.book_id).first()
            if existing_rating:
                # Update existing rating
                existing_rating.rating = user_rating
                existing_rating.rating_date = datetime.now().date()
                messagebox.showinfo("Success", "Rating updated successfully.")
                logging.info(f"Updated rating for book_id {book.book_id} by user_id {self.user.user_id}.")
            else:
                # Create new rating
                new_rating = Rating(
                    user_id=self.user.user_id,
                    book_id=book.book_id,
                    rating=user_rating,
                    rating_date=datetime.now().date()
                )
                session.add(new_rating)
                messagebox.showinfo("Success", "Rating submitted successfully.")
                logging.info(f"Submitted new rating for book_id {book.book_id} by user_id {self.user.user_id}.")
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to submit rating: {e}")
            messagebox.showerror("Error", f"Failed to submit rating: {e}")
        finally:
            session.close()
            # Refresh the My Ratings tab
            self.populate_my_ratings()

    def evaluate_model(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        # Check if CF model was trained
        if self.recommender_system.cf_recommender.algo is None:
            messagebox.showerror("Error",
                                 "Collaborative Filtering model is not trained due to insufficient ratings data.")
            return

        # Disable the button to prevent multiple clicks
        self.run_evaluation_button.config(state='disabled')
        self.update_status("Evaluating the model. Please wait...")
        self.evaluation_result_text.configure(state='normal')
        self.evaluation_result_text.delete(1.0, tk.END)
        self.evaluation_result_text.insert(tk.END, "Evaluating the Collaborative Filtering model...\n")
        self.evaluation_result_text.configure(state='disabled')

        # Run evaluation in a separate thread
        threading.Thread(target=self.run_evaluation, daemon=True).start()

    def run_evaluation(self):
        try:
            rmse = self.recommender_system.cf_recommender.evaluate_model()
            if rmse is not None:
                self.queue.put(lambda: self.display_evaluation_result(rmse))
            else:
                self.queue.put(lambda: messagebox.showerror("Error", "Failed to evaluate the model."))
        except Exception as e:
            self.queue.put(lambda: messagebox.showerror("Error", f"Evaluation failed: {e}"))

    def display_evaluation_result(self, rmse):
        self.evaluation_result_text.configure(state='normal')
        self.evaluation_result_text.delete(1.0, tk.END)
        self.evaluation_result_text.insert(tk.END, f"Collaborative Filtering Model RMSE: {rmse:.4f}")
        self.evaluation_result_text.configure(state='disabled')
        self.update_status("Model evaluation completed.")
        messagebox.showinfo("Model Evaluation", f"Collaborative Filtering Model RMSE: {rmse:.4f}")
        # Re-enable the button
        self.run_evaluation_button.config(state='normal')

    def update_profile(self):
        name = self.profile_name_entry.get().strip()
        email = self.profile_email_entry.get().strip()
        password = self.profile_password_entry.get().strip()
        confirm_password = self.profile_confirm_password_entry.get().strip()

        if not name or not email:
            messagebox.showerror("Error", "Name and Email fields cannot be empty.")
            return

        if password:
            if password != confirm_password:
                messagebox.showerror("Error", "Passwords do not match.")
                return
            hashed_password = hash_password(password).decode('utf-8')
        else:
            hashed_password = self.user.password  # Keep existing password

        # Connect to database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        try:
            # Check if email is being changed to one that already exists
            if email != self.user.email:
                existing_user = session.query(User).filter_by(email=email).first()
                if existing_user:
                    messagebox.showerror("Error", "Email already in use by another account.")
                    logging.warning(f"Attempted to update email to an existing one: {email}")
                    session.close()
                    return

            # Update user information
            user = session.query(User).filter_by(user_id=self.user.user_id).first()
            user.name = name
            user.email = email
            user.password = hashed_password
            session.commit()
            logging.info(f"User profile updated for user_id {self.user.user_id}.")
            messagebox.showinfo("Success", "Profile updated successfully.")
            self.update_status("Profile updated successfully.")
            # Update the user object
            self.user.name = name
            self.user.email = email
            self.user.password = hashed_password
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to update profile: {e}")
            messagebox.showerror("Error", f"Failed to update profile: {e}")
        finally:
            session.close()

    def populate_my_ratings(self):
        # Clear existing entries
        for row in self.my_ratings_tree.get_children():
            self.my_ratings_tree.delete(row)

        # Connect to database
        session = setup_database()
        if session is None:
            logging.error("Database connection failed.")
            return

        try:
            ratings = session.query(Rating).filter_by(user_id=self.user.user_id).options(joinedload(Rating.book)).all()
            logging.info(f"Loaded {len(ratings)} ratings for user_id {self.user.user_id}.")
        except Exception as e:
            logging.error(f"Failed to load user ratings: {e}")
            session.close()
            return
        session.close()

        if not ratings:
            self.my_ratings_tree.insert("", tk.END, values=("No ratings submitted yet.", "", ""))
            return

        for rating in ratings:
            if rating.book:
                book_title = rating.book.title
            else:
                book_title = "Unknown Book"
            rating_value = rating.rating
            rating_date = rating.rating_date.strftime("%Y-%m-%d") if rating.rating_date else "N/A"
            self.my_ratings_tree.insert("", tk.END, values=(book_title, rating_value, rating_date))

    def create_charts_tab(self):
        # Chart for Book Popularity
        chart_frame = ttk.Frame(self.charts_tab, padding="20 20 20 20")
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # Label
        ttk.Label(chart_frame, text="📈 Book Popularity Chart", font=('Helvetica', 14, 'bold')).pack(pady=10)

        # Placeholder for the chart
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button to Generate Chart
        self.generate_chart_button = ttk.Button(chart_frame, text="Generate Chart", command=self.generate_chart)
        self.generate_chart_button.pack(pady=10, ipadx=10, ipady=5)
        self.add_button_hover_effect(self.generate_chart_button)

    def generate_chart(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        # Fetch data for the chart
        try:
            books_df = self.recommender_system.books_df
            if 'borrow_count' not in books_df.columns:
                messagebox.showerror("Error", "'borrow_count' column not found in books data.")
                return
            top_books = books_df.sort_values(by='borrow_count', ascending=False).head(10)
            if top_books.empty:
                messagebox.showinfo("Info", "No data available to generate the chart.")
                return
            self.ax.clear()
            self.ax.barh(top_books['title'], top_books['borrow_count'], color='skyblue')
            self.ax.invert_yaxis()  # Highest at the top
            self.ax.set_xlabel('Borrow Count')
            self.ax.set_title('Top 10 Most Borrowed Books')
            for i, v in enumerate(top_books['borrow_count']):
                self.ax.text(v + 1, i, str(v), color='blue', va='center')
            self.chart_canvas.draw()
            self.update_status("Book popularity chart generated.")
            logging.info("Book popularity chart generated.")
        except Exception as e:
            logging.error(f"Failed to generate chart: {e}")
            messagebox.showerror("Error", f"Failed to generate chart: {e}")

    def initialize_system(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            # Disable the button to prevent multiple clicks
            self.init_button.config(state='disabled')
            self.update_status("Initializing the recommendation system. Please wait...")
            self.progress.start(10)  # Start the progress bar

            # Clear previous logs
            self.init_log_text.configure(state='normal')
            self.init_log_text.delete(1.0, tk.END)
            self.init_log_text.configure(state='disabled')

            # Run initialization in a separate daemon thread to keep the GUI responsive
            threading.Thread(target=self.run_initialization, daemon=True).start()
        else:
            messagebox.showwarning("Warning", "Recommendation system is already initialized.")

    def run_initialization(self):
        try:
            self.recommender_system, _, _, _ = initialize_recommender_system(message_queue=self.queue)
            # Put the callback in the queue
            self.queue.put(self.post_initialization)
        except Exception as e:
            # Handle exceptions and notify the main thread
            self.queue.put(lambda: messagebox.showerror("Error", f"Initialization failed: {e}"))

    def post_initialization(self):
        self.progress.stop()
        if self.recommender_system:
            self.update_status("Recommendation system initialized successfully.")
            messagebox.showinfo("Success", "Recommendation system initialized successfully.")

            # Enable Evaluate Model tab if CF model is trained
            # Tabs are 0-indexed: Evaluate Model is the 4th tab (after Initialize, Recommend, New User, Submit Rating)
            if self.recommender_system.cf_recommender.algo is not None:
                self.notebook.tab(4, state='normal')
            else:
                self.notebook.tab(4, state='disabled')
                messagebox.showwarning("Warning",
                                       "Collaborative Filtering model was not trained due to insufficient ratings data.")

            # Enable Charts tab
            self.notebook.tab(6, state='normal')
        else:
            self.update_status("Failed to initialize the recommendation system.")
            messagebox.showerror("Error", "Failed to initialize the recommendation system.")
        # Re-enable the button
        self.init_button.config(state='normal')

    def get_recommendations(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        book_title = self.book_title_entry.get().strip()
        top_n = self.recommend_top_n.get()

        if not book_title:
            messagebox.showerror("Error", "Book title cannot be empty.")
            return

        recommendations = self.recommender_system.get_recommendations(
            user_id=self.user.user_id,
            book_title=book_title,
            top_n=top_n
        )
        self.recommend_text.configure(state='normal')
        self.recommend_text.delete(1.0, tk.END)
        if recommendations:
            self.recommend_text.insert(tk.END, f"📚 Recommended Books based on '{book_title}':\n\n")
            for idx, book in enumerate(recommendations, 1):
                self.recommend_text.insert(tk.END, f"{idx}. {book}\n")
        else:
            self.recommend_text.insert(tk.END, "No recommendations found for the entered book title.")
            logging.info(f"'{book_title}' not found in the database or no similar books available.")
        self.recommend_text.configure(state='disabled')

    def get_new_user_recommendations(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        top_n = self.new_user_top_n.get()
        recommendations = self.recommender_system.recommend_for_new_user(
            books_df=self.recommender_system.books_df,
            top_n=top_n
        )
        self.new_user_recs_text.configure(state='normal')
        self.new_user_recs_text.delete(1.0, tk.END)
        if recommendations:
            self.new_user_recs_text.insert(tk.END, "📚 Recommended Books for New Users:\n\n")
            for idx, book in enumerate(recommendations, 1):
                self.new_user_recs_text.insert(tk.END, f"{idx}. {book}\n")
        else:
            self.new_user_recs_text.insert(tk.END, "No recommendations found.")
        self.new_user_recs_text.configure(state='disabled')

    def submit_rating(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        book_title = self.submit_book_title_entry.get().strip()
        user_rating = self.submit_rating_var.get()

        if not book_title:
            messagebox.showerror("Error", "Book title cannot be empty.")
            return

        # Fetch book_id from title
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        book = session.query(Book).filter_by(title=book_title).first()
        session.close()

        if not book:
            messagebox.showerror("Error", f"Book titled '{book_title}' not found.")
            return

        # Insert rating into database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        try:
            # Check if user has already rated this book
            existing_rating = session.query(Rating).filter_by(user_id=self.user.user_id, book_id=book.book_id).first()
            if existing_rating:
                # Update existing rating
                existing_rating.rating = user_rating
                existing_rating.rating_date = datetime.now().date()
                messagebox.showinfo("Success", "Rating updated successfully.")
                logging.info(f"Updated rating for book_id {book.book_id} by user_id {self.user.user_id}.")
            else:
                # Create new rating
                new_rating = Rating(
                    user_id=self.user.user_id,
                    book_id=book.book_id,
                    rating=user_rating,
                    rating_date=datetime.now().date()
                )
                session.add(new_rating)
                messagebox.showinfo("Success", "Rating submitted successfully.")
                logging.info(f"Submitted new rating for book_id {book.book_id} by user_id {self.user.user_id}.")
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to submit rating: {e}")
            messagebox.showerror("Error", f"Failed to submit rating: {e}")
        finally:
            session.close()
            # Refresh the My Ratings tab
            self.populate_my_ratings()

    def evaluate_model(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        # Check if CF model was trained
        if self.recommender_system.cf_recommender.algo is None:
            messagebox.showerror("Error",
                                 "Collaborative Filtering model is not trained due to insufficient ratings data.")
            return

        # Disable the button to prevent multiple clicks
        self.run_evaluation_button.config(state='disabled')
        self.update_status("Evaluating the model. Please wait...")
        self.evaluation_result_text.configure(state='normal')
        self.evaluation_result_text.delete(1.0, tk.END)
        self.evaluation_result_text.insert(tk.END, "Evaluating the Collaborative Filtering model...\n")
        self.evaluation_result_text.configure(state='disabled')

        # Run evaluation in a separate thread
        threading.Thread(target=self.run_evaluation, daemon=True).start()

    def run_evaluation(self):
        try:
            rmse = self.recommender_system.cf_recommender.evaluate_model()
            if rmse is not None:
                self.queue.put(lambda: self.display_evaluation_result(rmse))
            else:
                self.queue.put(lambda: messagebox.showerror("Error", "Failed to evaluate the model."))
        except Exception as e:
            self.queue.put(lambda: messagebox.showerror("Error", f"Evaluation failed: {e}"))

    def display_evaluation_result(self, rmse):
        self.evaluation_result_text.configure(state='normal')
        self.evaluation_result_text.delete(1.0, tk.END)
        self.evaluation_result_text.insert(tk.END, f"Collaborative Filtering Model RMSE: {rmse:.4f}")
        self.evaluation_result_text.configure(state='disabled')
        self.update_status("Model evaluation completed.")
        messagebox.showinfo("Model Evaluation", f"Collaborative Filtering Model RMSE: {rmse:.4f}")
        # Re-enable the button
        self.run_evaluation_button.config(state='normal')

    def update_profile(self):
        name = self.profile_name_entry.get().strip()
        email = self.profile_email_entry.get().strip()
        password = self.profile_password_entry.get().strip()
        confirm_password = self.profile_confirm_password_entry.get().strip()

        if not name or not email:
            messagebox.showerror("Error", "Name and Email fields cannot be empty.")
            return

        if password:
            if password != confirm_password:
                messagebox.showerror("Error", "Passwords do not match.")
                return
            hashed_password = hash_password(password).decode('utf-8')
        else:
            hashed_password = self.user.password  # Keep existing password

        # Connect to database
        session = setup_database()
        if session is None:
            messagebox.showerror("Error", "Database connection failed.")
            return

        try:
            # Check if email is being changed to one that already exists
            if email != self.user.email:
                existing_user = session.query(User).filter_by(email=email).first()
                if existing_user:
                    messagebox.showerror("Error", "Email already in use by another account.")
                    logging.warning(f"Attempted to update email to an existing one: {email}")
                    session.close()
                    return

            # Update user information
            user = session.query(User).filter_by(user_id=self.user.user_id).first()
            user.name = name
            user.email = email
            user.password = hashed_password
            session.commit()
            logging.info(f"User profile updated for user_id {self.user.user_id}.")
            messagebox.showinfo("Success", "Profile updated successfully.")
            self.update_status("Profile updated successfully.")
            # Update the user object
            self.user.name = name
            self.user.email = email
            self.user.password = hashed_password
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to update profile: {e}")
            messagebox.showerror("Error", f"Failed to update profile: {e}")
        finally:
            session.close()

    def populate_my_ratings(self):
        # Clear existing entries
        for row in self.my_ratings_tree.get_children():
            self.my_ratings_tree.delete(row)

        # Connect to database
        session = setup_database()
        if session is None:
            logging.error("Database connection failed.")
            return

        try:
            ratings = session.query(Rating).filter_by(user_id=self.user.user_id).options(joinedload(Rating.book)).all()
            logging.info(f"Loaded {len(ratings)} ratings for user_id {self.user.user_id}.")
        except Exception as e:
            logging.error(f"Failed to load user ratings: {e}")
            session.close()
            return
        session.close()

        if not ratings:
            self.my_ratings_tree.insert("", tk.END, values=("No ratings submitted yet.", "", ""))
            return

        for rating in ratings:
            if rating.book:
                book_title = rating.book.title
            else:
                book_title = "Unknown Book"
            rating_value = rating.rating
            rating_date = rating.rating_date.strftime("%Y-%m-%d") if rating.rating_date else "N/A"
            self.my_ratings_tree.insert("", tk.END, values=(book_title, rating_value, rating_date))

    def generate_chart(self):
        if not hasattr(self, 'recommender_system') or self.recommender_system is None:
            messagebox.showerror("Error", "Recommendation system not initialized.")
            return

        # Fetch data for the chart
        try:
            books_df = self.recommender_system.books_df
            if 'borrow_count' not in books_df.columns:
                messagebox.showerror("Error", "'borrow_count' column not found in books data.")
                return
            top_books = books_df.sort_values(by='borrow_count', ascending=False).head(10)
            if top_books.empty:
                messagebox.showinfo("Info", "No data available to generate the chart.")
                return
            self.ax.clear()
            self.ax.barh(top_books['title'], top_books['borrow_count'], color='skyblue')
            self.ax.invert_yaxis()  # Highest at the top
            self.ax.set_xlabel('Borrow Count')
            self.ax.set_title('Top 10 Most Borrowed Books')
            for i, v in enumerate(top_books['borrow_count']):
                self.ax.text(v + 1, i, str(v), color='blue', va='center')
            self.chart_canvas.draw()
            self.update_status("Book popularity chart generated.")
            logging.info("Book popularity chart generated.")
        except Exception as e:
            logging.error(f"Failed to generate chart: {e}")
            messagebox.showerror("Error", f"Failed to generate chart: {e}")

    def logout(self):
        confirm = messagebox.askyesno("Logout", "Are you sure you want to logout?")
        if confirm:
            # Close the main app window
            self.master.destroy()
            # Reopen the login window
            self.app.start_login_window()


# ------------------------------
# Main Application Class to Manage Windows
# ------------------------------

class LibraryRecommenderApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        self.login_window = None
        self.register_window = None
        self.main_app_window = None
        self.start_login_window()

    def start_login_window(self):
        self.login_window = tk.Toplevel(self.root)
        self.login_window.protocol("WM_DELETE_WINDOW", self.root.quit)  # Ensure proper closure
        LoginWindow(self.login_window, self)

    def open_register_window(self):
        if self.register_window and tk.Toplevel.winfo_exists(self.register_window):
            self.register_window.focus()
            return
        self.register_window = tk.Toplevel(self.root)
        self.register_window.protocol("WM_DELETE_WINDOW", self.register_window.destroy)
        RegisterWindow(self.register_window, self)

    def open_main_app(self, user):
        if self.main_app_window and tk.Toplevel.winfo_exists(self.main_app_window):
            self.main_app_window.focus()
            return
        self.main_app_window = tk.Toplevel(self.root)
        self.main_app_window.protocol("WM_DELETE_WINDOW", self.root.quit)  # Ensure proper closure
        MainApp(self.main_app_window, user)
        # Close the login window
        self.login_window.destroy()

    def run(self):
        self.root.mainloop()

# ------------------------------
# Main Function to Launch the Application
# ------------------------------

def main():
    app = LibraryRecommenderApp()
    app.run()

if __name__ == "__main__":
    main()
