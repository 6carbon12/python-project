from sklearn.svm import SVC
from pandas import read_csv
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt

# Get the data
iris_data = read_csv('data/iris_dataset_cleaned.csv')
x = iris_data.drop(columns=['species_index', 'species'])
y = iris_data['species_index']

# Test Train split of 30% and 70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the model
model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Get accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of model: {accuracy*100:.2f}")

with open('reports/accuracy_svm.txt', 'w') as f:
    f.write(f"{accuracy*100:.2f}%")

# Get confusion matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Setosa', 'Versicolor', 'Virginica'])

# Plot and save
disp.plot(cmap='Greens')
plt.title('Confusion Matrix')
plt.savefig('reports/confusion_matrix_svm.png')
print("Confusion Matrix saved to 'reports/confusion_matrix_svm.png'")

# Save the model to disk so it can be loaded later
dump(model, 'models/model_svm.pkl')
print("Model saved to models/model_svm.pkl")
