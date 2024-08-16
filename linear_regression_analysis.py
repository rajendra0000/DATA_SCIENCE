# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'iris' with your own dataset if needed)
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(df.head())

# Define the feature (X) and target (y)
X = df[['sepal_length']]  # Feature: sepal length (replace with your feature)
y = df['petal_length']    # Target: petal length (replace with your target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the regression line and actual vs. predicted values
plt.figure(figsize=(10, 6))

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plot the test data
plt.scatter(X_test, y_test, color='green', label='Test Data')

# Plot the regression line
X_range = np.linspace(X['sepal_length'].min(), X['sepal_length'].max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.plot(X_range, y_range, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Linear Regression: Sepal Length vs Petal Length')
plt.legend()
plt.show()
