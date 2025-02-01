import pandas as pd
import streamlit as st
import pickle

# Load dataset
cars_df = pd.read_csv("cars24-car-price.csv")

# Page Title
st.title("🚗 Cars24 Used Car Price Prediction")

st.subheader("Dataset Sample")
st.dataframe(cars_df.head())  # Show sample data

# Encoding Dictionary
encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "seller_type": {"Dealer": 1, "Individual": 2, "Trustmark Dealer": 3},
    "transmission_type": {"Manual": 1, "Automatic": 2}
}

# Function to Predict Car Price
def model_pred(fuel_type, transmission_type, engine, seats):
    try:
        with open("car_pred", "rb") as file:
            reg_model = pickle.load(file)

        input_features = [[2018.0, 1, 40000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]
        prediction = reg_model.predict(input_features)
        return round(prediction[0], 2)  # Round prediction for better readability

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# User Input Columns
col1, col2 = st.columns(2)

fuel_type = col1.selectbox("Select Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
engine = col1.slider("Set the Engine Power (CC)", 500, 5000, 1500)  # Default value improved

transmission_type = col2.selectbox("Select Transmission Type", ["Manual", "Automatic"])
seats = col2.number_input("Enter the Number of Seats", min_value=4, max_value=7, step=1)

# Prediction Button
if st.button("🔍 Predict Price"):
    fuel_type_encoded = encode_dict['fuel_type'][fuel_type]
    transmission_type_encoded = encode_dict['transmission_type'][transmission_type]
    
    price = model_pred(fuel_type_encoded, transmission_type_encoded, engine, seats)
    
    if price:
        # Display price in lakhs with "Lakh" suffix
        st.success(f"💰 Predicted Price of the Car: ₹ {price} Lakh")
        st.metric(label="Predicted Car Price", value=f"₹ {price} Lakh")  