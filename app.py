
import streamlit as st
import pandas as pd
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as file:
        return pickle.load(file)

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
    "Enter the property details below to estimate its selling price."
)

st.info(
    "This model uses housing data from Ames, Iowa, USA. "
    "Predicted prices are in US dollars ($)."
)

# User inputs
st.subheader("Enter House Details")

gr_liv_area = st.number_input(
    "Living Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1800,
    step=50
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

# Neighborhood dropdown
neighborhoods = [
    "Blmngtn",
    "Blueste",
    "BrDale",
    "BrkSide",
    "ClearCr",
    "CollgCr",
    "Crawfor",
    "Edwards",
    "Gilbert",
    "IDOTRR",
    "MeadowV",
    "Mitchel",
    "NAmes",
    "NoRidge",
    "NPkVill",
    "NridgHt",
    "NWAmes",
    "OldTown",
    "SWISU",
    "Sawyer",
    "SawyerW",
    "Somerst",
    "StoneBr",
    "Timber",
    "Veenker"
]

neighborhood = st.selectbox(
    "Select Neighborhood",
    options=neighborhoods,
    index=neighborhoods.index("NAmes"),
    help="Choose a neighborhood from the Ames, Iowa housing dataset."
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

st.caption(
    "Select a neighborhood from the list. "
    "These options belong to the US dataset, not Indian cities."
)

# Prediction
if st.button("Predict House Price", type="primary"):
    house = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": neighborhood,
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
            "This is a machine learning estimate, "
            "not a guaranteed market price."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
