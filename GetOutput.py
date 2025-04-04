import tkinter as tk
from tkinter import filedialog

def get_folder_path():
    # Create a Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Open the folder explorer dialog to select a folder
    folder_path = filedialog.askdirectory(title="Select Folder to Save File")
    
    if folder_path:  # If a folder was selected
        return folder_path
    else:
        return None  # If no folder was selected

# Example usage
#folder_path = get_folder_path()
#if folder_path:
#    print(f"Selected folder: {folder_path}")
#else:
#    print("No folder was selected.")

