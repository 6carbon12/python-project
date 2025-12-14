# Iris Species Predictor

A Machine Learning application that predicts Iris flower species based on measurements. It a Random Forest model with a dark-mode GUI.

## 🛠️ Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Directories**
    Ensure the following folders exist in your project root:
    * `data/` (Place your `iris_dataset.csv` here)
    * `models/`
    * `reports/`

## 🚀 How to Run

### 1. Clean the Data
Preprocesses the raw CSV to ensure only valid features are used.
```bash
python src/clean.py
````

### 2. Train the model
Train the model on the cleaned data and get the confusion matrix and accuracy
```bash
python src/model.py
````

### 3. Launch the App

Opens the GUI to predict species.
```bash
python main.py
```
