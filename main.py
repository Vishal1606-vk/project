import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
# Assuming the data is in a CSV file named 'advertising.csv'
data = pd.read_csv('Sales.csv')

# Display the first few rows of the dataset
print(data.head())

# Separate the features (TV, Radio, Newspaper) and the target variable (Sales)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting the actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")
plt.show()

# Display the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Function to predict sales based on new advertising expenditure
def predict_sales(tv, radio, newspaper):
    return model.predict([[tv, radio, newspaper]])

# Example prediction
new_tv = 150
new_radio = 20
new_newspaper = 30
predicted_sales = predict_sales(new_tv, new_radio, new_newspaper)
print(f'Predicted Sales for TV: {new_tv}, Radio: {new_radio}, Newspaper: {new_newspaper} -> {predicted_sales[0]}')
