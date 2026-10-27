
import streamlit as st
import pandas as pd
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# App title
st.title("🏠 House Price Prediction")
st.write(
    "Enter the house details below to estimate its selling price."
)

st.info("This model was trained on US housing data. Prices are predicted in US dollars ($).")

# User inputs
st.subheader("Enter House Details")

gr_liv_area = st.number_input(
    "Living Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1800
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

neighborhood = st.text_input(
    "Neighborhood Code",
    value="NAmes",
    help="Enter a neighborhood code found in the training dataset, such as NAmes."
)

overall_qual = st.slider(
    "Overall Quality (1–10)",
    min_value=1,
    max_value=10,
    value=7
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2005,
    step=1
)

# Prediction button
if st.button("Predict House Price", type="primary"):
    if not neighborhood.strip():
        st.error("Please enter a neighborhood code.")
    else:
        house = pd.DataFrame([{
            "GrLivArea": gr_liv_area,
            "BedroomAbvGr": bedrooms,
            "Neighborhood": neighborhood.strip(),
            "OverallQual": overall_qual,
            "YearBuilt": year_built
        }])

        try:
            predicted_price = model.predict(house)[0]

            st.success("Prediction completed!")
            st.metric(
                label="Estimated House Selling Price",
                value=f"${predicted_price:,.2f}"
            )
            st.caption(
                "This is a machine learning estimate, not a guaranteed market price."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
