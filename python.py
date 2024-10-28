import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import simpledialog
import threading

# Load movies from CSV into a pandas DataFrame
def load_movies(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        messagebox.showerror("File Error", "CSV file not found!")
        return pd.DataFrame()

# Change this line to the correct path
movies = load_movies(r"C:\xampp\htdocs\speechapp\speechapp\movies.csv")

# Function to animate text
def animate_text(label, text, delay=100):
    def update_text(i=0):
        if i < len(text):
            label.config(text=text[:i+1])
            root.after(delay, update_text, i+1)
    update_text()

# Function to recommend movies with a simulated process time
def recommend_movies():
    # Display a custom message box while processing
    processing_popup = tk.Toplevel(root)
    processing_popup.title("Processing")
    processing_popup.geometry("300x100")
    processing_popup.configure(bg="#202020")
    tk.Label(processing_popup, text="Fetching the best movies for you...", bg="#202020", fg="white", font=("Helvetica", 12)).pack(pady=20)
    root.after(2000, processing_popup.destroy)

    # Run the recommendation process in a separate thread to prevent UI freezing
    threading.Thread(target=process_recommendation).start()

# Function to process movie recommendations
def process_recommendation():
    # Get user inputs
    genre = genre_combobox.get()
    min_year = year_entry.get()
    min_rating = rating_combobox.get()

    # Check if inputs are valid
    try:
        min_year = int(min_year)
    except ValueError:
        min_year = 0  # Default to 0 if not provided

    try:
        min_rating_value = int(min_rating)
    except ValueError:
        min_rating_value = 0  # Default to 0 if not provided

    # Apply expert system rules to filter the movies
    filtered_movies = movies[
        (movies['Genre'].str.contains(genre, case=False, na=False)) &
        (movies['Release Year'] >= min_year) &
        (movies['Rating'].round() == min_rating_value)
    ]

    # Display results in the listbox
    movie_listbox.delete(0, tk.END)  # Clear the previous recommendations
    if not filtered_movies.empty:
        for _, movie in filtered_movies.iterrows():
            movie_listbox.insert(tk.END, f"{movie['Title']} ({movie['Release Year']}) - {movie['Rating']}/10")
    else:
        movie_listbox.insert(tk.END, "Sorry, no movies found matching your criteria. Please try different options!")
        user_message = simpledialog.askstring("Feedback", "No results found. Would you like to provide feedback?")
        if user_message:
            messagebox.showinfo("Feedback Received", "Thank you for your feedback!")

# Setup the main window
root = tk.Tk()
root.title("YouTube-Inspired Movie Recommender System")
root.geometry("1400x900")
root.configure(bg="#121212")  # Updated to an even darker background for a sleeker look  # Set background color to resemble YouTube's dark mode

# Create a header frame for a banner
header_frame = tk.Frame(root, bg="#FF0000", height=150, relief="ridge", bd=10)  # Added a border for better contrast  # YouTube red color
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="", bg="#FF0000", fg="white", font=("Helvetica", 36, "bold italic"))  # Updated font for a more modern look
header_label.pack(pady=20)
animate_text(header_label, "📺 Welcome to the Movie Recommender System 🎥", delay=50)

# Create a frame for user inputs
input_frame = tk.Frame(root, bg="#1E1E1E", padx=40, pady=40, relief="ridge", bd=8)  # Darkened the frame color and added relief for a 3D effect  # Dark gray background
input_frame.pack(pady=20, fill="x", padx=20)

# Labels and input fields
style = ttk.Style()
style.configure("TLabel", background="#202020", foreground="#f5f5f5", font=("Helvetica", 16))  # Updated color scheme and font

# Genre Selection
ttk.Label(input_frame, text="Select Genre:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
genre_combobox = ttk.Combobox(input_frame, values=list(movies['Genre'].unique()), width=35, font=("Helvetica", 14), state="readonly")  # Made the dropdown read-only for better UX
genre_combobox.grid(row=0, column=1, padx=10, pady=10)

genre_combobox.set("Select Genre")

# Year Input
ttk.Label(input_frame, text="Minimum Release Year:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
year_entry = ttk.Entry(input_frame, width=35, font=("Helvetica", 14))  # Increased entry width for better alignment
year_entry.grid(row=1, column=1, padx=10, pady=10)

year_entry.insert(0, "1970")

# Rating Selection
ttk.Label(input_frame, text="Minimum Rating:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
rating_combobox = ttk.Combobox(input_frame, values=list(range(1, 11)), width=35, font=("Helvetica", 14), state="readonly")
rating_combobox.grid(row=2, column=1, padx=10, pady=10)

rating_combobox.set("Select Rating (1-10)")

# Recommend Button with a custom style
recommend_button = tk.Button(input_frame, text="Recommend Movies", command=recommend_movies,
                             bg="#FF4500", fg="white", font=("Helvetica", 20, "bold"),
                             activebackground="#FF6347", activeforeground="white", relief="ridge", bd=10, cursor="hand2")  # Enhanced button style for a more engaging appearance
recommend_button.grid(row=3, column=0, columnspan=2, pady=20)

# Processing Label
process_label = tk.Label(input_frame, text="", bg="#282828", fg="white", font=("Arial", 12, "italic"))
process_label.grid(row=4, column=0, columnspan=2)

# Create a frame for displaying recommendations
output_frame = tk.Frame(root, bg="#1E1E1E", padx=40, pady=40, relief="ridge", bd=10)  # Updated frame style for more visual depth
output_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Listbox to show movie recommendations with improved styling
movie_listbox = tk.Listbox(output_frame, width=90, height=20, bg="#181818", fg="#ffffff", font=("Helvetica", 16),
                           relief="flat", selectbackground="#FF4500", selectforeground="white", highlightthickness=2, highlightbackground="#FF4500")  # Updated colors and added border for listbox
movie_listbox.pack(pady=10, fill="both", expand=True)

# Add scrollbar to the listbox
scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=movie_listbox.yview)
scrollbar.pack(side="right", fill="y")
movie_listbox.config(yscrollcommand=scrollbar.set)

# Add a footer frame for additional information or credits
footer_frame = tk.Frame(root, bg="#FF0000", height=80, relief="ridge", bd=6)  # Added a sunken border for emphasis
footer_frame.pack(fill="x", side="bottom")

footer_label = tk.Label(footer_frame, text="Powered by Eduard's Movie Recommender System", bg="#FF0000",
                        fg="white", font=("Helvetica", 18, "bold italic"))  # Updated font for consistency with the new design
footer_label.pack(pady=10)

# Run the main loop
root.mainloop()