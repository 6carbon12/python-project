import pandas as pd

# Keep only the needed columns
iris = pd.read_csv('data/iris_dataset.csv')
required_colums = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]
iris_cleaned = iris[required_colums].copy()

# Convert speices into specis_index
mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris_cleaned['species_index'] = iris_cleaned['species'].map(mapping)

# Save this cleaned dataset into a new file
iris_cleaned.to_csv('data/iris_dataset_cleaned.csv', index=False)
