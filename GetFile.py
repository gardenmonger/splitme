import tkinter as tk
from tkinter import filedialog

def get_file_path():
    # Create a Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Open the file explorer dialog and ask for a file
    file_path = filedialog.askopenfilename(title="Select a file")
    
    if file_path:  # If a file was selected
        print(f"Selected file: {file_path}")
        return file_path
    else:
        return None  # If no file was selected
