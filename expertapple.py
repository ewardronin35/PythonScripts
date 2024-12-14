import tkinter as tk
from tkinter import ttk, messagebox
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, Date
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Database setup
Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'
    book_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)
    genre = Column(String, nullable=False)
    borrow_count = Column(Integer, default=0)

class Rating(Base):
    __tablename__ = 'ratings'
    rating_id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer)
    user_id = Column(Integer)
    rating = Column(Float)

def setup_database():
    engine = create_engine('sqlite:///library.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# Tkinter Application
class LibraryRecommenderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Library Recommender System")
        self.master.geometry("900x600")

        self.session = setup_database()

        # Tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        # Recommendation Tab
        self.recommend_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recommend_tab, text="Get Recommendations")

        # New User Recommendation Tab
        self.new_user_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.new_user_tab, text="New User Recommendations")

        # Submit Rating Tab
        self.submit_rating_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.submit_rating_tab, text="Submit Rating")

        # Initialize Tabs
        self.create_recommend_tab()
        self.create_new_user_tab()
        self.create_submit_rating_tab()

    def create_recommend_tab(self):
        ttk.Label(self.recommend_tab, text="Enter Book Title:").pack(pady=10)
        self.book_entry = ttk.Entry(self.recommend_tab, width=50)
        self.book_entry.pack(pady=10)

        ttk.Button(self.recommend_tab, text="Get Recommendations", command=self.get_recommendations).pack(pady=10)

        self.recommend_text = tk.Text(self.recommend_tab, wrap=tk.WORD, height=15, width=80)
        self.recommend_text.pack(pady=10)

    def create_new_user_tab(self):
        ttk.Label(self.new_user_tab, text="Top N Recommendations:").pack(pady=10)
        self.new_user_spinbox = ttk.Spinbox(self.new_user_tab, from_=1, to=20, width=5)
        self.new_user_spinbox.pack(pady=10)

        ttk.Button(self.new_user_tab, text="Get New User Recommendations", command=self.get_new_user_recommendations).pack(pady=10)

        self.new_user_text = tk.Text(self.new_user_tab, wrap=tk.WORD, height=15, width=80)
        self.new_user_text.pack(pady=10)

    def create_submit_rating_tab(self):
        ttk.Label(self.submit_rating_tab, text="Enter Book Title:").pack(pady=10)
        self.submit_book_entry = ttk.Entry(self.submit_rating_tab, width=50)
        self.submit_book_entry.pack(pady=10)

        ttk.Label(self.submit_rating_tab, text="Enter Rating (1-5):").pack(pady=10)
        self.rating_spinbox = ttk.Spinbox(self.submit_rating_tab, from_=1, to=5, increment=0.5, width=5)
        self.rating_spinbox.pack(pady=10)

        ttk.Button(self.submit_rating_tab, text="Submit Rating", command=self.submit_rating).pack(pady=10)

    def get_recommendations(self):
        book_title = self.book_entry.get().strip()
        if not book_title:
            messagebox.showerror("Error", "Please enter a book title.")
            return

        # Fetch the entered book from the database
        entered_book = self.session.query(Book).filter(func.lower(Book.title) == book_title.lower()).first()

        if not entered_book:
            messagebox.showerror("Error", f"Book '{book_title}' not found in the database.")
            return

        # Debug: Print the entered book's details
        print(f"Entered Book: {entered_book.title} by {entered_book.author}, Genre: {entered_book.genre}")

        # Fetch books in the same genre, excluding the entered book
        recommendations = self.session.query(Book).filter(
            func.lower(Book.genre) == entered_book.genre.lower(),
            Book.book_id != entered_book.book_id
        ).all()

        # Display recommendations
        self.recommend_text.delete(1.0, tk.END)
        if recommendations:
            self.recommend_text.insert(tk.END, f"Recommended Books in Genre '{entered_book.genre}':\n")
            for idx, book in enumerate(recommendations, start=1):
                self.recommend_text.insert(tk.END, f"{idx}. {book.title} by {book.author}\n")
        else:
            self.recommend_text.insert(tk.END, "No recommendations found in the same genre.")

    def get_new_user_recommendations(self):
        top_n = int(self.new_user_spinbox.get())
        books = self.session.query(Book).order_by(Book.borrow_count.desc()).limit(top_n).all()

        self.new_user_text.delete(1.0, tk.END)
        if books:
            self.new_user_text.insert(tk.END, "Top Recommendations for New Users:\n")
            for idx, book in enumerate(books, start=1):
                self.new_user_text.insert(tk.END, f"{idx}. {book.title} (Borrowed {book.borrow_count} times)\n")
        else:
            self.new_user_text.insert(tk.END, "No recommendations available.")

    def submit_rating(self):
        book_title = self.submit_book_entry.get().strip()
        rating = float(self.rating_spinbox.get())

        if not book_title:
            messagebox.showerror("Error", "Please enter a book title.")
            return

        book = self.session.query(Book).filter_by(title=book_title).first()
        if not book:
            messagebox.showerror("Error", f"Book '{book_title}' not found.")
            return

        new_rating = Rating(book_id=book.book_id, user_id=1, rating=rating)  # Default user_id = 1 for simplicity
        self.session.add(new_rating)
        self.session.commit()

        messagebox.showinfo("Success", f"Rating for '{book_title}' submitted successfully.")

# Main Application
def main():
    root = tk.Tk()
    app = LibraryRecommenderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()