from pathlib import Path
import pickle as pk

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

base_dir = Path(__file__).resolve().parent
model = pk.load(open(base_dir / "model.pkl", "rb"))

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

car_data = pd.read_csv(base_dir / "Cardetails.csv")


def model_name(carName):
    carName = carName.split()[0]
    return carName


car_data["name"] = car_data["name"].apply(model_name)

encoders = {
    "fuel": LabelEncoder().fit(car_data["fuel"]),
    "seller_type": LabelEncoder().fit(car_data["seller_type"]),
    "name": LabelEncoder().fit(car_data["name"]),
    "owner": LabelEncoder().fit(car_data["owner"]),
    "transmission": LabelEncoder().fit(car_data["transmission"]),
}

car_name = st.selectbox("Select Car Brand", sorted(car_data["name"].unique()))

mf_year = st.slider("Car Manufactured year", 1994, 2026, 2010)
km = st.slider("No. of KMs Driven", 10, 1000000, 500000)
fuel = st.selectbox("Fuel type", ["Diesel", "Petrol", "CNG", "LPG"])
seller = st.selectbox("Seller type", ["Individual", "Dealer", "TrustMark Dealer"])

trans = st.selectbox("Transmission Type", ["Automatic", "Manual"])

Owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Others"])
mil = st.slider("Car mileage", 10, 50, 25)
eng = st.slider("Engine CC", 700, 4000, 2000)
max_p = st.slider("Max Power", 0, 400, 200)
seats = st.slider("No. of seats", 2, 10, 4)

pred = st.button("Predict")


if pred:
    @st.dialog("🚗 Car details")
    def show_details():
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c1:
            with st.container(border=True):
                st.caption("Car Model")
                st.subheader(car_name)
            with st.container(border=True):  
                st.caption("Manufactured Year ")
                st.subheader(mf_year)
            with st.container(border=True):    
                st.caption("Car mileage")
                st.subheader(mil)
            with st.container(border=True):
                st.caption("Total Seats")
                st.subheader(seats)
        with c2:
            with st.container(border=True):
                st.caption("Fuel type")
                st.subheader(fuel)
            with st.container(border=True):
                st.caption("No. of KMs Driven")
                st.subheader(km)
            with st.container(border=True):
                st.caption("Engine")
                st.subheader(eng)
        with c3:
            with st.container(border=True):
                st.caption("Seller type")
                st.subheader(seller)
            with st.container(border=True):
                st.caption("Owner type")
                st.subheader(Owner)
            with st.container(border=True):
                st.caption("Max Power")
                st.subheader(max_p)

        input_car_data = pd.DataFrame(
            {
                "name": [car_name],
                "year": [mf_year],
                "km_driven": [km],
                "fuel": [fuel],
                "seller_type": [seller],
                "transmission": [trans],
                "owner": [Owner],
                "mileage": [mil],
                "engine": [eng],
                "max_power": [max_p],
                "seats": [seats],
            }
        )

        for column, encoder in encoders.items():
            input_car_data[column] = encoder.transform(input_car_data[column].astype(str))

        car_price = model.predict(input_car_data)
        predicted_price = car_price[0]

        st.success(f"💰 Predicted Car Price: ₹ {predicted_price:,.2f}")

    show_details()
