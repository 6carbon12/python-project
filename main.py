import tkinter as tk
from tkinter import ttk, messagebox
from joblib import load
import pandas as pd
import os

# --- Constants for Styling ---
BG_COLOR = "#2b2b2b"       # Dark Grey Background
FG_COLOR = "#ffffff"       # White Text
ACCENT_COLOR = "#3498db"   # Blue Accent for buttons
ENTRY_BG = "#3d3d3d"       # Slightly lighter grey for inputs


class IrisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # 1. Window Setup
        self.title("Iris Species Predictor")
        self.geometry("400x480")
        self.configure(bg=BG_COLOR)

        # Load Model
        self.model = self.load_model()
        self.species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

        # 2. Apply Custom Styles (Dark Theme)
        self.setup_styles()

        # 3. UI Layout
        self.create_widgets()

    def load_model(self):
        path = 'models/iris_model.pkl'
        if not os.path.exists(path):
            messagebox.showerror(
                "Error", "Model not found!")
            self.destroy()
            return None
        return load(path)

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')  # An easier to configure theme

        # Configure Frame, Label, and Button styles
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel",
                        background=BG_COLOR,
                        foreground=FG_COLOR,
                        font=("Helvetica", 11))
        style.configure("Header.TLabel",
                        font=("Helvetica", 18, "bold"),
                        padding=10)

        # Custom Button Style
        style.configure("TButton",
                        font=("Helvetica", 11, "bold"),
                        background=ACCENT_COLOR,
                        foreground=FG_COLOR,
                        borderwidth=0,
                        focuscolor=BG_COLOR)
        # Darker blue on hover
        style.map("TButton", background=[('active', '#2980b9')])

    def create_widgets(self):
        # -- Title --
        title = ttk.Label(self, text="Flower Analysis", style="Header.TLabel")
        title.pack(pady=(20, 10))

        # -- Input Container (Centered) --
        input_frame = ttk.Frame(self)
        input_frame.pack(padx=30, pady=10, fill="x")

        # Create Inputs
        self.entry_sl = self.create_input(input_frame, "Sepal Length (cm):", 0)
        self.entry_sw = self.create_input(input_frame, "Sepal Width (cm):", 1)
        self.entry_pl = self.create_input(input_frame, "Petal Length (cm):", 2)
        self.entry_pw = self.create_input(input_frame, "Petal Width (cm):", 3)

        # -- Predict Button --
        # Using a Frame to add external padding to the button
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=25, fill="x", padx=40)

        self.btn_predict = ttk.Button(btn_frame,
                                      text="PREDICT SPECIES",
                                      command=self.predict_species)
        self.btn_predict.pack(ipady=5, fill="x")

        # -- Result Label --
        self.lbl_result = tk.Label(self, text="Result: Waiting...",
                                   font=("Helvetica", 14),
                                   bg=BG_COLOR,
                                   fg="#888888")  # Grey text initially
        self.lbl_result.pack(pady=10)

    def create_input(self, parent, label_text, row):
        # Helper to create a Label + Entry pair
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=0, sticky="w", pady=8)

        # UPDATED: Added 'highlightcolor' to change the border when focused
        entry = tk.Entry(parent,
                         bg=ENTRY_BG,
                         fg=FG_COLOR,
                         insertbackground="white",  # Color of the blinking cursor
                         relief="flat",
                         highlightthickness=1,
                         highlightbackground="#555",   # Border color when inactive
                         # Border color when active (Blue)
                         highlightcolor=ACCENT_COLOR)

        entry.grid(row=row, column=1, sticky="e", pady=8, ipadx=5, ipady=3)
        return entry

    def predict_species(self):
        try:
            # Get values
            data = [
                float(self.entry_sl.get()),
                float(self.entry_sw.get()),
                float(self.entry_pl.get()),
                float(self.entry_pw.get())
            ]

            # Predict
            cols = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']
            df = pd.DataFrame([data], columns=cols)
            prediction_index = self.model.predict(df)[0]
            species = self.species_mapping[prediction_index]

            # Update UI with colors
            color_map = {'Setosa': '#2ecc71',
                         'Versicolor': '#3498db',
                         'Virginica': '#e74c3c'}
            self.lbl_result.config(
                text=f"Result: {species.upper()}", fg=color_map[species])

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")


if __name__ == "__main__":
    app = IrisApp()
    app.mainloop()
