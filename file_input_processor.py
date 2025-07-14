import tkinter as tk
from tkinter import filedialog

def upload_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        print(f"chosen file: {filepath}")
        return filepath

window = tk.Tk()
window.title("AI Lesson Plan Adapter")

upload_button = tk.Button(root, text="Upload File", command = upload_file)
upload_button.pack(pady=10)

file = upload_file()
if file:
    with open(file, "rb") as f:
        file_data = f.read()
print(type(file_data))