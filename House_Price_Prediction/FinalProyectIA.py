#This project predicts house prices based on features such as location, size, number of bedrooms, and age using a linear regression model. A Tkinter-based GUI is provided for user interaction.

#Example Use Case
#Enter inputs such as location, size (sqft), number of bedrooms, and age.
#Click "Predict" to view the estimated house price.
#Results
#The model's performance metrics:

#Mean Absolute Error (MAE): XX,XXX (depends on dataset split)
#Root Mean Square Error (RMSE): XX,XXX
#Predicted values for sample input:

#Example result from inputs: Location: Urban, Size: 1800 sqft, Bedrooms: 3, Age: 10 years. Predicted Price: $375,000

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import messagebox

# Sample dataset
data = {
    'Location': ['Downtown', 'Suburb', 'Rural', 'Urban', 'Coastal'],
    'Size (sqft)': [1200, 1800, 2400, 1500, 2000],
    'Bedrooms': [3, 4, 5, 3, 4],
    'Age (years)': [10, 5, 15, 7, 8],
    'Price ($)': [350000, 420000, 300000, 375000, 450000]
}

# Data conversion into a pandas DataFrame
df = pd.DataFrame(data)

# Data Preprocessing: One-hot encode 'Location'
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop('Price ($)', axis=1)
y = df['Price ($)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}, RMSE: {rmse}")

# Define the GUI
def predict_price():
    try:
        # Get user inputs
        location = location_entry.get()
        size = float(size_entry.get())
        bedrooms = int(bedrooms_entry.get())
        age = int(age_entry.get())

        # Validate location
        if location not in ['Downtown', 'Suburb', 'Rural', 'Urban', 'Coastal']:
            raise ValueError("Invalid location. Choose from: Downtown, Suburb, Rural, Urban, Coastal.")

        # Prepare the input data to match the trained model
        input_data = pd.DataFrame({
            'Size (sqft)': [size],
            'Bedrooms': [bedrooms],
            'Age (years)': [age],
            'Location_Coastal': [1 if location == 'Coastal' else 0],
            'Location_Rural': [1 if location == 'Rural' else 0],
            'Location_Suburb': [1 if location == 'Suburb' else 0],
            'Location_Urban': [1 if location == 'Urban' else 0]
            # No column for 'Location_Downtown' because it was dropped in training (drop_first=True)
        })

        # Ensure the columns match the training data
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)[0]
        result_label.config(text=f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        # Show error in a message box
        messagebox.showerror("Error", str(e))


# Create the main window
root = tk.Tk()
root.title("House Price Prediction")

# Define variables
location_var = tk.StringVar()
size_var = tk.StringVar()
bedrooms_var = tk.StringVar()
age_var = tk.StringVar()

# Create and place widgets
tk.Label(root, text="Location (Downtown/Suburb/Rural/Urban/Coastal):").grid(row=0, column=0, padx=10, pady=5)
location_entry = tk.Entry(root, textvariable=location_var)
location_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Size (sqft):").grid(row=1, column=0, padx=10, pady=5)
size_entry = tk.Entry(root, textvariable=size_var)
size_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Bedrooms:").grid(row=2, column=0, padx=10, pady=5)
bedrooms_entry = tk.Entry(root, textvariable=bedrooms_var)
bedrooms_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Age (years):").grid(row=3, column=0, padx=10, pady=5)
age_entry = tk.Entry(root, textvariable=age_var)
age_entry.grid(row=3, column=1, padx=10, pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

# Label to display the prediction result
result_label = tk.Label(root, text="")
result_label.grid(row=5, column=0, columnspan=2, pady=10)


# Run the application
root.mainloop()