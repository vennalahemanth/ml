import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Title for the Streamlit app
st.title("Total Revenue Prediction App")

# Load the dataset (Make sure to upload the dataset here or provide a path)
@st.cache_data  # Caching the data for performance
def load_data():
     # Adjust if needed
    return pd.read_excel(r"C:\Users\heman\OneDrive\Desktop\WEEK-4\Online Sales Data.xlsx")

sales_data = load_data()

# Display first few rows of data in Streamlit app
st.write("### Sample Data")
st.write(sales_data.head())

# Step 1: Data Preprocessing

# Selecting features and target
X = sales_data.drop(columns=["Transaction ID", "Date", "Total Revenue"])  # Dropping ID, Date, and Total Revenue columns
y = sales_data["Total Revenue"]  # Target column is now Total Revenue

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define columns for numeric and categorical preprocessing
numeric_features = ["Units Sold", "Unit Price"]
categorical_features = ["Product Category", "Product Name", "Region", "Payment Method"]

# Numeric Preprocessing: Handle missing values using mean imputation
numeric_transformer = SimpleImputer(strategy="mean")

# Categorical Preprocessing: Handle missing values and one-hot encode categorical variables
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values with the most frequent
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode the categorical features
])

# Combine both numeric and categorical preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),  # Numeric preprocessing
        ("cat", categorical_transformer, categorical_features)  # Categorical preprocessing
    ])

# Step 2: Model Building
# Define the model pipeline, which includes preprocessing and the Random Forest regressor
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),  # Preprocess both numeric and categorical data
    ("regressor", RandomForestRegressor(random_state=42))  # Use Random Forest for regression
])

# Step 3: Model Training
# Train the model on the training set
model_pipeline.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Features")

# Create input widgets for the user to provide input
units_sold = st.sidebar.number_input("Enter Units Sold", min_value=0.0, value=100.0)
unit_price = st.sidebar.number_input("Enter Unit Price", min_value=0.0, value=20.0)
product_category = st.sidebar.selectbox("Select Product Category", sales_data["Product Category"].unique())
product_name = st.sidebar.selectbox("Select Product Name", sales_data["Product Name"].unique())
region = st.sidebar.selectbox("Select Region", sales_data["Region"].unique())
payment_method = st.sidebar.selectbox("Select Payment Method", sales_data["Payment Method"].unique())

# Function to predict Total Revenue from user input
def predict_total_revenue(units_sold, unit_price, product_category, product_name, region, payment_method):
    # Create a dataframe for the new input data
    input_data = pd.DataFrame({
        "Units Sold": [units_sold],
        "Unit Price": [unit_price],
        "Product Category": [product_category],
        "Product Name": [product_name],
        "Region": [region],
        "Payment Method": [payment_method]
    })

    # Use the trained model pipeline to predict Total Revenue
    predicted_total_revenue = model_pipeline.predict(input_data)
    return predicted_total_revenue[0]

# Button to trigger prediction
if st.sidebar.button("Predict Total Revenue"):
    # Make the prediction
    predicted_revenue = predict_total_revenue(units_sold, unit_price, product_category, product_name, region, payment_method)
    
    # Display the predicted result
    st.write(f"### Predicted Total Revenue: ${predicted_revenue:.2f}")

