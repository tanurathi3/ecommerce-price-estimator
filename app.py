import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.title("E-Commerce Product Price Estimator")
st.write("Enter product details to estimate its price.")

df = pd.read_csv("ecommerce_scraped_data.csv")

features = ["category", "brand", "rating", "variants", "stock", "discount_percent"]

X = df[features]
y = df["price"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "brand"])
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

category = st.selectbox("Category", sorted(df["category"].unique()))
brand = st.selectbox("Brand", sorted(df["brand"].unique()))
rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
variants = st.number_input("Number of Variants", 1, 100, 3)
stock = st.number_input("Stock", 0, 10000, 50)
discount = st.slider("Discount (%)", 0, 100, 10)

if st.button("Estimate Price"):
    data = pd.DataFrame([{
        "category": category,
        "brand": brand,
        "rating": rating,
        "variants": variants,
        "stock": stock,
        "discount_percent": discount
    }])

    price = model.predict(data)[0]
    st.success(f"Estimated Price: ₹{price:.2f}")
