import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from joblib import load
import pandas as pd
import os
import subprocess

# --- Constants for Styling ---
BG_COLOR = "#2b2b2b"
FG_COLOR = "#ffffff"
ACCENT_COLOR = "#3498db"
ENTRY_BG = "#3d3d3d"
PANEL_BG = "#333333"


class IrisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # 1. Window Setup
        self.title("Iris Species Workbench")
        self.geometry("950x800")
        self.configure(bg=BG_COLOR)

        # 2. State & Data
        self.species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        self.models = {}
        self.model_files = {
            'Random Forest': 'models/model_rf.pkl',
            'Decision Tree': 'models/model_dtc.pkl',
            'SVM': 'models/model_svm.pkl',
            'KNN': 'models/model_knn.pkl'
        }
        # Map Display Name -> Script Filename (for retraining)
        self.script_files = {
            'Random Forest': 'src/model/random_forest.py',
            'Decision Tree': 'src/model/decision_tree_classifier.py',
            'SVM': 'src/model/support_vector_machines.py',
            'KNN': 'src/model/k_nearest_neighbours.py'
        }

        self.load_all_models()

        # 3. Apply Styles
        self.setup_styles()

        # 4. Create Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.tab_predict = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_predict, text='  Predict  ')
        self.notebook.add(self.tab_train, text='  Train & Evaluate  ')

        # 5. Build UI
        self.build_predict_tab()
        self.build_train_tab()

    def load_all_models(self):
        print("Loading models...")
        for name, path in self.model_files.items():
            if os.path.exists(path):
                try:
                    self.models[name] = load(path)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
            else:
                print(f"Warning: {path} not found.")

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR,
                        foreground=FG_COLOR, font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("SubHeader.TLabel", font=(
            "Helvetica", 12, "bold"), foreground="#aaaaaa")

        # Notebook (Tab) Styles
        style.configure("TNotebook", background=BG_COLOR)
        style.configure("TNotebook.Tab", background="#444",
                        foreground="white")
        style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)])

        # Button Styles
        style.configure("TButton", font=("Helvetica", 10, "bold"),
                        background=ACCENT_COLOR, foreground="white", borderwidth=0)
        style.map("TButton", background=[('active', '#2980b9')])

        # Action Button (Red) for retraining
        style.configure("Action.TButton", background="#e74c3c")
        style.map("Action.TButton", background=[('active', '#c0392b')])

    # ==========================================
    #               PREDICTION TAB
    # ==========================================
    def build_predict_tab(self):
        predict_tab = ttk.Frame(self.tab_predict)
        predict_tab.pack(fill='both', expand=True, padx=20, pady=20)

        # -- Header --
        ttk.Label(predict_tab, text="Classify New Flower",
                  style="Header.TLabel").pack(pady=(0, 20))

        # -- Model Selection --
        sel_frame = ttk.Frame(predict_tab)
        sel_frame.pack(pady=10, anchor="center")
        ttk.Label(sel_frame, text="Select Model:",
                  style="SubHeader.TLabel").pack(side='left')

        self.model_var = tk.StringVar(value="All Models (Ensemble)")
        options = ["All Models (Ensemble)"] + list(self.model_files.keys())
        self.combo_model = ttk.Combobox(
            sel_frame, textvariable=self.model_var, values=options, state="readonly", font=("Helvetica", 11))
        self.combo_model.pack(side='right',
                              padx=(10, 0))

        # -- Inputs --
        input_frame = ttk.Frame(predict_tab, style="TFrame")
        input_frame.pack(pady=20)

        self.entries = {}
        labels = ["Sepal Length (cm)", "Sepal Width (cm)",
                  "Petal Length (cm)", "Petal Width (cm)"]
        for i, text in enumerate(labels):
            ttk.Label(input_frame, text=text).grid(
                row=i, column=0, sticky="w", pady=8, padx=10)
            entry = tk.Entry(input_frame, bg=ENTRY_BG, fg=FG_COLOR, insertbackground="white",
                             relief="flat", highlightthickness=1, highlightcolor=ACCENT_COLOR)
            entry.grid(row=i, column=1, sticky="e", pady=8, ipady=3)
            self.entries[i] = entry

        # -- Predict Button --
        btn = ttk.Button(predict_tab, text="PREDICT SPECIES",
                         command=self.run_prediction)
        btn.pack(fill='x', pady=20, ipady=5)

        # -- Results Section --
        self.res_frame = tk.Frame(
            predict_tab, bg=PANEL_BG, bd=0, highlightthickness=0)
        self.res_frame.pack(fill='both', expand=True, pady=10)

        self.lbl_main_result = tk.Label(self.res_frame, text="Ready", font=(
            "Helvetica", 16, "bold"), bg=PANEL_BG, fg="#888")
        self.lbl_main_result.pack(pady=10)

        # Details / Confidence
        self.lbl_details = tk.Label(self.res_frame, text="", font=(
            "Courier", 10), bg=PANEL_BG, fg="#ccc", justify="left")
        self.lbl_details.pack(pady=5, padx=10, anchor="w")

    def run_prediction(self):
        try:
            data = []
            constraints = [
                (0.5, 12.0, "Sepal Length"),
                (0.5, 10.0, "Sepal Width"),
                (0.1, 10.0, "Petal Length"),
                (0.1, 5.0,  "Petal Width")
            ]

            # 1. Gather & Validate Data
            for i in range(4):
                val_str = self.entries[i].get().strip()

                # Check for empty inputs
                if not val_str:
                    messagebox.showwarning("Missing Input", f"Please enter a value for {
                                           constraints[i][2]}.")
                    return

                val = float(val_str)
                min_v, max_v, name = constraints[i]

                # The Guard Rail: Check if value is realistic
                if not (min_v <= val <= max_v):
                    messagebox.showwarning(
                        "Unrealistic Value",
                        f"{name} ({val} cm) is impossible for an Iris flower.\n\n"
                        f"Please enter a value between {min_v} and {max_v} cm."
                    )
                    return  # Stop processing immediately

                data.append(val)

            # 2. Proceed if all checks pass
            df = pd.DataFrame([data], columns=['sepal_length',
                              'sepal_width', 'petal_length', 'petal_width'])

            selected_option = self.model_var.get()

            species, details = None, None
            if selected_option == "All Models (Ensemble)":
                species, details = self.predict_ensemble(df)
            else:
                species, details = self.predict_single(selected_option, df)

            self.display_result(species, details)

        except ValueError:
            messagebox.showerror(
                "Input Error", "Please enter valid numbers (e.g. 5.1).")
        except Exception as e:
            messagebox.showerror(
                "Error", f"An unexpected error occurred:\n{e}")

    def predict_single(self, model_name, df):
        model = self.models.get(model_name)
        if not model:
            self.lbl_main_result.config(text="Model not loaded!", fg="red")
            return

        pred_idx = model.predict(df)[0]
        species = self.species_mapping[pred_idx]

        # Get Probability if available
        details = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = proba[pred_idx] * 100
            details = f"Confidence: {
                confidence:.2f}%\n\nProbability Breakdown:\n"
            details += f"  Setosa:     {proba[0]*100:.1f}%\n"
            details += f"  Versicolor: {proba[1]*100:.1f}%\n"
            details += f"  Virginica:  {proba[2]*100:.1f}%"

        return (species, details)

    def predict_ensemble(self, df):
        votes = []
        details = "Model Votes:\n"

        for name, model in self.models.items():
            pred_idx = model.predict(df)[0]
            species = self.species_mapping[pred_idx]
            votes.append(species)

            conf_str = ""
            if hasattr(model, "predict_proba"):
                conf = max(model.predict_proba(df)[0]) * 100
                conf_str = f"({conf:.1f}%)"

            details += f"  • {name}: {species} {conf_str}\n"

        # Majority Vote
        final_species = max(set(votes), key=votes.count)

        return (final_species, details)

    def display_result(self, species, details_text):
        color_map = {'Setosa': '#2ecc71',
                     'Versicolor': '#3498db', 'Virginica': '#e74c3c'}
        self.lbl_main_result.config(
            text=species.upper(), fg=color_map.get(species, FG_COLOR))
        self.lbl_details.config(text=details_text)

    # ==========================================
    #                 TRAIN TAB
    # ==========================================
    def build_train_tab(self):
        # Split into Left (List) and Right (Image)
        container = ttk.Frame(self.tab_train)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        left_pane = ttk.Frame(container)
        left_pane.pack(side='left', fill='y', padx=(0, 20))

        right_pane = ttk.Frame(container, style="TFrame")
        right_pane.pack(side='right', fill='both', expand=True)

        ttk.Label(left_pane, text="Model Management",
                  style="Header.TLabel").pack(pady=(0, 20), anchor='w')

        # Create row for each model
        for name, script in self.script_files.items():
            row = ttk.Frame(left_pane, style="TFrame")
            row.pack(fill='x', pady=10)

            lbl = ttk.Label(row, text=name, width=15,
                            font=("Helvetica", 11, "bold"))
            lbl.pack(side='left')

            # Train Button
            btn = ttk.Button(row, text="Retrain", width=8, style="Action.TButton",
                             command=lambda s=script, n=name: self.train_model(s, n, report_frame=right_pane))
            btn.pack(side='left', padx=10)

            # View Button
            view_btn = ttk.Button(row, text="View Report", width=12,
                                  command=lambda n=name: self.show_report(n, right_pane))
            view_btn.pack(side='left')

        # Placeholder for Image
        self.img_label = tk.Label(right_pane, text="Select a model to view Confusion Matrix",
                                  bg="#222", fg="#666", font=("Helvetica", 12))
        self.img_label.pack(fill='both', expand=True)

        # Placeholder for accuracy
        self.accuracy_label = tk.Label(right_pane, text="",
                                    bg=BG_COLOR, fg="#2ecc71", font=("Helvetica", 16, "bold"))
        self.accuracy_label.pack(side='bottom', pady=10)

    def train_model(self, script_name, model_name, report_frame=None):
        if not os.path.exists(script_name):
            messagebox.showerror("Error", f"Script {script_name} not found!")
            return

        try:
            # Run the script
            subprocess.run(["python", script_name], check=True)
            messagebox.showinfo(
                "Success", f"{model_name} retrained successfully!")

            # Reload the model in memory
            self.load_all_models()

            if report_frame:
                self.show_report(model_name, report_frame)

        except subprocess.CalledProcessError as e:
            messagebox.showerror(
                "Training Failed", f"Error running script:\n{e}")

    def show_report(self, model_name, parent_frame):
        file_map = {
            'Random Forest': ('reports/confusion_matrix_rf.png', 'reports/accuracy_rf.txt'),
            'Decision Tree': ('reports/confusion_matrix_dtc.png', 'reports/accuracy_dtc.txt'),
            'SVM':           ('reports/confusion_matrix_svm.png', 'reports/accuracy_svm.txt'),
            'KNN':           ('reports/confusion_matrix_knn.png', 'reports/accuracy_knn.txt')
        }

        img_path, acc_path = file_map.get(model_name, (None, None))

        # 1. Load Image
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            # Resize nicely
            img = img.resize((450, 350))
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
        else:
            self.img_label.config(image="", text="Report image not found.\nTry retraining.")

        # 2. Load Accuracy Text
        if acc_path and os.path.exists(acc_path):
            with open(acc_path, 'r') as f:
                acc_value = f.read().strip()
            self.accuracy_label.config(text=f"{model_name} Model Accuracy: {acc_value}")
        else:
            self.accuracy_label.config(text="")


if __name__ == "__main__":
    app = IrisApp()
    app.mainloop()
