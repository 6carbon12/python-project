from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

iris_data = read_csv('data/iris_dataset_cleaned.csv')
sns.pairplot(iris_data.drop(columns=['species_index']),
             hue='species',
             diag_kind='kde')

plt.show()
plt.savefig('reports/pair-plot.png')
