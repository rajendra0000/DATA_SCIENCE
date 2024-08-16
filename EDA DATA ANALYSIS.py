import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = sns.load_dataset('iris')

# Display basic information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Histograms of all numerical features
df.hist(figsize=(10, 10))
plt.tight_layout()
plt.show()

# Scatter plot for two features
plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
