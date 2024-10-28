import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load CSV File Function
def load_csv():
    file_path = "customer_sales.csv"  # assuming the CSV file is in the same directory as your script
    if file_path:
        try:
            global df
            df = pd.read_csv(file_path)
            display_data(df)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")

# Display Data in Treeview
def display_data(data):
    clear_tree()
    tree["column"] = list(data.columns)
    tree["show"] = "headings"
    for col in tree["column"]:
        tree.heading(col, text=col)
        tree.column(col, width=150)
    for row in data.itertuples(index=False):
        tree.insert("", "end", values=row)

# Clear Treeview
def clear_tree():
    tree.delete(*tree.get_children())

# Inspect Data Types and Missing Values
def inspect_data():
    if 'df' not in globals():
        messagebox.showerror("Error", "No dataset loaded.")
        return
    missing_values = df.isnull().sum()
    data_types = df.dtypes
    messagebox.showinfo("Data Info", f"Data Types:\n{data_types}\n\nMissing Values:\n{missing_values}")

# Handle Missing Values
def handle_missing():
    if 'df' not in globals():
        messagebox.showerror("Error", "No dataset loaded.")
        return
    global df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    messagebox.showinfo("Info", "Missing values in numeric columns have been filled with median values.")
    display_data(df)

# Data Cleaning and Transformation
def clean_data():
    if 'df' not in globals():
        messagebox.showerror("Error", "No dataset loaded.")
        return
    global df
    df.drop_duplicates(inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    messagebox.showinfo("Info", "Data cleaning and transformation completed.")
    display_data(df)

# Exploratory Data Analysis (EDA)
def eda():
    if 'df' not in globals():
        messagebox.showerror("Error", "No dataset loaded.")
        return
    plt.figure(figsize=(12, 6))
    sns.histplot(df.select_dtypes(include=[np.number]), kde=True)
    plt.title("Numeric Data Distribution")
    plt.show()

# Reporting Insights
def report_insights():
    if 'df' not in globals():
        messagebox.showerror("Error", "No dataset loaded.")
        return
    numeric_cols = df.select_dtypes(include=[np.number])
    correlations = numeric_cols.corr()
    trend = correlations.unstack().sort_values(ascending=False).drop_duplicates()
    messagebox.showinfo("Insights", f"Trends and Correlations:\n{trend.head()}\n\nOutliers should be reviewed visually.")

# GUI Design
root = tk.Tk()
root.title("Expert System Lab Activity - Finals #2")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#ffffff", pady=10, padx=10)
frame.pack(pady=20)

load_button = tk.Button(frame, text="Load CSV File", command=load_csv, bg="#4caf50", fg="white", font=("Poppins", 12))
load_button.grid(row=0, column=0, padx=5, pady=5)

inspect_button = tk.Button(frame, text="Inspect Data", command=inspect_data, bg="#2196f3", fg="white", font=("Poppins", 12))
inspect_button.grid(row=0, column=1, padx=5, pady=5)

handle_missing_button = tk.Button(frame, text="Handle Missing Values", command=handle_missing, bg="#ff9800", fg="white", font=("Poppins", 12))
handle_missing_button.grid(row=0, column=2, padx=5, pady=5)

clean_button = tk.Button(frame, text="Clean Data", command=clean_data, bg="#9c27b0", fg="white", font=("Poppins", 12))
clean_button.grid(row=0, column=3, padx=5, pady=5)

eda_button = tk.Button(frame, text="EDA", command=eda, bg="#f44336", fg="white", font=("Poppins", 12))
eda_button.grid(row=0, column=4, padx=5, pady=5)

report_button = tk.Button(frame, text="Report Insights", command=report_insights, bg="#3f51b5", fg="white", font=("Poppins", 12))
report_button.grid(row=0, column=5, padx=5, pady=5)

# Treeview for Data Display
tree_frame = tk.Frame(root)
tree_frame.pack(pady=20)

tree = ttk.Treeview(tree_frame, show='headings')
tree.pack()

# Scrollbar for Treeview
scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
tree.configure(xscrollcommand=scrollbar.set)
scrollbar.pack(fill='x')

root.mainloop()
