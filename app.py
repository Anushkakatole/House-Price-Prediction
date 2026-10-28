
import streamlit as st
import pandas as pd
import pickle
import sklearn
from pathlib import Path

# Page configuration — must come before other Streamlit commands
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load trained model safely
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "house_price_model.pkl"

    with open(model_path, "rb") as file:
        return pickle.load(file)


# App title
st.title("🏠 House Price Prediction")
st.write("Enter the property details below to estimate its selling price.")

st.info(
    "This model uses housing data from Ames, Iowa, USA. "
    "Predicted prices are in US dollars ($)."
)

# Display model-loading errors clearly
try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the house price prediction model.")
    st.warning(
        "The saved model may be incompatible with the installed "
        "scikit-learn version. Please regenerate the model using "
        "a compatible environment."
    )

    with st.expander("Technical details"):
        st.code(
            f"Error: {type(error).__name__}: {error}\n"
            f"Installed scikit-learn: {sklearn.__version__}"
        )

    st.stop()


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

neighborhoods = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill",
    "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer",
    "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"
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
    "These neighborhoods belong to the US housing dataset, "
    "not Indian cities."
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
        st.error("Prediction failed. Please check the model inputs.")

        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
