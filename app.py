import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model = joblib.load(
    "models/real_estate_price_model.pkl"
)

df = pd.read_csv(
    "data/final_housing.csv"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🏠 Real Estate Price Predictor & Investment Analyzer"
)

st.write(
    "A Machine Learning based system for property price "
    "prediction and investment analysis."
)

st.markdown("---")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Price Predictor",
        "💰 Investment Analyzer",
        "📊 Model Performance",
        "📍 Location Analysis"
    ]
)


# =========================================================
# TAB 1 — PRICE PREDICTOR
# =========================================================

with tab1:

    st.header("🏠 Property Price Prediction")

    st.write(
        "Enter the property details below to estimate "
        "its market price."
    )

    col1, col2 = st.columns(2)

    with col1:

        locations = sorted(
            df["location"]
            .dropna()
            .unique()
        )

        location = st.selectbox(
            "📍 Location",
            locations
        )

        total_sqft = st.number_input(
            "📐 Area (sq.ft)",
            min_value=300.0,
            max_value=20000.0,
            value=1200.0,
            step=50.0
        )

        bhk = st.number_input(
            "🛏️ Bedrooms (BHK)",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:

        bath = st.number_input(
            "🚿 Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        balcony = st.number_input(
            "🌇 Balconies",
            min_value=0,
            max_value=5,
            value=1
        )

        area_type = st.selectbox(
            "🏢 Area Type",
            sorted(df["area_type"].unique())
        )


    if st.button(
        "🔮 Predict Property Price",
        key="predict"
    ):

        total_rooms = bhk + bath

        input_data = pd.DataFrame(
            {
                "area_type": [area_type],
                "location": [location],
                "total_sqft": [total_sqft],
                "bath": [bath],
                "balcony": [balcony],
                "bhk": [bhk],
                "total_rooms": [total_rooms]
            }
        )

        prediction = model.predict(
            input_data
        )[0]

        st.markdown("---")

        st.subheader(
            "💰 Estimated Property Price"
        )

        st.success(
            f"₹ {prediction:.2f} Lakhs"
        )

        st.info(
            f"Estimated price for a {bhk} BHK property "
            f"with {total_sqft:.0f} sq.ft in {location}."
        )


# =========================================================
# TAB 2 — INVESTMENT ANALYZER
# =========================================================

with tab2:

    st.header(
        "💰 Property Investment Analyzer"
    )

    st.write(
        "Compare the listing price with the Machine "
        "Learning estimated price."
    )

    col1, col2 = st.columns(2)

    with col1:

        locations = sorted(
            df["location"]
            .dropna()
            .unique()
        )

        inv_location = st.selectbox(
            "📍 Location",
            locations,
            key="investment_location"
        )

        inv_area = st.number_input(
            "📐 Area (sq.ft)",
            min_value=300.0,
            max_value=20000.0,
            value=1200.0,
            step=50.0,
            key="investment_area"
        )

        inv_bhk = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            max_value=10,
            value=2,
            key="investment_bhk"
        )

    with col2:

        inv_bath = st.number_input(
            "🚿 Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            key="investment_bath"
        )

        inv_balcony = st.number_input(
            "🌇 Balconies",
            min_value=0,
            max_value=5,
            value=1,
            key="investment_balcony"
        )

        inv_area_type = st.selectbox(
            "🏢 Area Type",
            sorted(df["area_type"].unique()),
            key="investment_area_type"
        )

        listing_price = st.number_input(
            "💵 Listing Price (₹ Lakhs)",
            min_value=1.0,
            max_value=5000.0,
            value=90.0,
            step=1.0
        )


    if st.button(
        "🔍 Analyze Investment",
        key="investment"
    ):

        total_rooms = (
            inv_bhk + inv_bath
        )

        input_data = pd.DataFrame(
            {
                "area_type": [inv_area_type],
                "location": [inv_location],
                "total_sqft": [inv_area],
                "bath": [inv_bath],
                "balcony": [inv_balcony],
                "bhk": [inv_bhk],
                "total_rooms": [total_rooms]
            }
        )

        prediction = model.predict(
            input_data
        )[0]

        difference = (
            prediction - listing_price
        )

        percentage_difference = (
            difference / prediction
        ) * 100


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "ML Estimated Price",
                f"₹ {prediction:.2f} L"
            )

        with c2:

            st.metric(
                "Listing Price",
                f"₹ {listing_price:.2f} L"
            )

        with c3:

            st.metric(
                "Difference",
                f"₹ {abs(difference):.2f} L"
            )


        # -------------------------------------------------
        # Investment Signal
        # -------------------------------------------------

        st.subheader(
            "🏷️ Investment Signal"
        )

        if percentage_difference >= 10:

            st.success(
                "🟢 Potentially Undervalued"
            )

            st.write(
                "The listing price is significantly below "
                "the model's estimated value."
            )

        elif percentage_difference <= -10:

            st.error(
                "🔴 Potentially Overvalued"
            )

            st.write(
                "The listing price is significantly above "
                "the model's estimated value."
            )

        else:

            st.warning(
                "🟡 Relatively Fair Price"
            )

            st.write(
                "The listing price is relatively close "
                "to the model's estimated value."
            )


        st.write(
            f"Price difference: "
            f"**{percentage_difference:.2f}%**"
        )


        # -------------------------------------------------
        # Price Per Square Foot
        # -------------------------------------------------

        st.subheader(
            "📐 Price Per Square Foot"
        )

        listing_pps = (
            listing_price * 100000
        ) / inv_area

        predicted_pps = (
            prediction * 100000
        ) / inv_area


        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Listing ₹/sq.ft",
                f"₹ {listing_pps:,.0f}"
            )

        with c2:

            st.metric(
                "Predicted ₹/sq.ft",
                f"₹ {predicted_pps:,.0f}"
            )


        # -------------------------------------------------
        # Price Comparison Chart
        # -------------------------------------------------

        st.subheader(
            "📊 Price Comparison"
        )

        chart_data = pd.DataFrame(
            {
                "Price Type": [
                    "ML Estimated",
                    "Listing Price"
                ],
                "Price (Lakhs)": [
                    prediction,
                    listing_price
                ]
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "Price Type"
            )
        )

        st.caption(
            "⚠️ The investment signal is a model-based "
            "screening indicator, not guaranteed financial advice."
        )


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================

with tab3:

    st.header(
        "📊 Machine Learning Model Performance"
    )

    st.write(
        "Performance of the improved Random Forest "
        "Regression model on the test dataset."
    )


    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R² Score",
            "0.7758"
        )

    with c2:

        st.metric(
            "MAE",
            "₹19.18 L"
        )

    with c3:

        st.metric(
            "RMSE",
            "₹39.34 L"
        )


    st.markdown("---")


    # -------------------------------------------------
    # Model comparison
    # -------------------------------------------------

    st.subheader(
        "🤖 Model Comparison"
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest",
                "Improved Random Forest"
            ],

            "MAE": [
                43.83,
                33.83,
                19.18
            ],

            "RMSE": [
                113.99,
                102.29,
                39.34
            ],

            "R²": [
                0.4393,
                0.5485,
                0.7758
            ]
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # -------------------------------------------------
    # Actual vs Predicted
    # -------------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Prices"
    )

    if os.path.exists(
        "actual_vs_predicted.png"
    ):

        st.image(
            "actual_vs_predicted.png",
            use_container_width=True
        )

    else:

        st.warning(
            "actual_vs_predicted.png not found. "
            "Run model_analysis.py first."
        )


    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    st.subheader(
        "🌳 Feature Importance"
    )

    if os.path.exists(
        "feature_importance.png"
    ):

        st.image(
            "feature_importance.png",
            use_container_width=True
        )

    else:

        st.warning(
            "feature_importance.png not found. "
            "Run model_analysis.py first."
        )


# =========================================================
# TAB 4 — LOCATION ANALYSIS
# =========================================================

with tab4:

    st.header(
        "📍 Location-Based Property Analysis"
    )

    locations = sorted(
        df["location"]
        .dropna()
        .unique()
    )

    selected_location = st.selectbox(
        "Select a Location",
        locations,
        key="location_analysis"
    )


    location_data = df[
        df["location"] == selected_location
    ]


    if len(location_data) > 0:

        median_price = (
            location_data["price"].median()
        )

        average_price = (
            location_data["price"].mean()
        )

        median_pps = (
            location_data["price_per_sqft"].median()
        )


        # -------------------------------------------------
        # Location Metrics
        # -------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Median Price",
                f"₹ {median_price:.2f} L"
            )

        with c2:

            st.metric(
                "Average Price",
                f"₹ {average_price:.2f} L"
            )

        with c3:

            st.metric(
                "Median ₹/sq.ft",
                f"₹ {median_pps:,.0f}"
            )


        st.markdown("---")


        # -------------------------------------------------
        # Location Scatter Plot
        # -------------------------------------------------

        st.subheader(
            f"📈 Price vs Area — {selected_location}"
        )

        location_chart = location_data[
            ["total_sqft", "price"]
        ].copy()

        location_chart = location_chart.sort_values(
            "total_sqft"
        )

        st.scatter_chart(
            location_chart,
            x="total_sqft",
            y="price"
        )


        st.caption(
            f"Based on {len(location_data)} properties "
            f"in the dataset."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "BCS586 Data Science Mini Project | "
    "Real Estate Price Predictor & Investment Analyzer"
)

st.caption(
    "⚠️ Predictions are estimates generated by a "
    "Machine Learning model and should not be treated "
    "as guaranteed property valuations or financial advice."
)