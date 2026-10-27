import streamlit as st
import requests
import pandas as pd
import io
import os

# Configure page style
st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Forecasting Service")
st.write("Deploys a containerized forecasting pipeline using our Tuned Random Forest model to predict outlet sales revenue.")
# Sidebar URL config
backend_url = st.sidebar.text_input(
    "Backend Service URL",
    value=os.getenv("BACKEND_URL", "http://backend-service:7860"),
    help="Change this if deploying on a custom domain or outside Docker network"
)

tab1, tab2 = st.tabs(["🎯 Single Record Prediction", "📁 Batch Prediction (CSV)"])

with tab1:
    st.header("Single Product Sales Prediction")
    st.write("Fill out the details below to generate a single-item sales forecast:")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.66)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=0.5, value=0.027, format="%.4f")
        product_mrp = st.number_input("Product MRP", min_value=10.0, max_value=300.0, value=117.08)
        product_type_category = st.selectbox("Product Type Category", [
            "Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood",
        ])

    with col2:
        # store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=50, value=17)

    if st.button("Forecast Sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Product_Type_Category": product_type_category,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city_type,
            "Store_Type": store_type,
            "Store_Age_Years": int(store_age)
        }

        with st.spinner("Calling prediction API..."):
            try:
                res = res = requests.post(
                    f"{backend_url.rstrip('/')}/v1/predict",
                    json=payload,
                    timeout=(5, 120),
                )
                if res.status_code == 200:
                    prediction = res.json().get('Product_Store_Sales_Total_Predicted')
                    st.success(f"Predicted Sales Revenue: **${prediction:,.2f}**")
                else:
                    st.error(f"Error: {res.json().get('error', 'Inference failed')}")
            except Exception as e:
                st.error(f"Could not reach the backend service: {e}")

with tab2:
    st.header("Batch Sales Prediction")
    st.write("Upload a CSV containing multiple records matching our dataset schema to receive batch forecasts:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the file to display a preview
        input_df = pd.read_csv(uploaded_file)
        st.write("**Data Preview (First 5 Rows):**")
        st.dataframe(input_df.head())

        if st.button("Run Batch Forecast", type="primary"):
            # Reset pointer and send file stream
            uploaded_file.seek(0)
            files = {'file': uploaded_file.getvalue()}

            with st.spinner("Processing batch on prediction server..."):
                try:
                    res = requests.post(
                        f"{backend_url.rstrip('/')}/v1/predictbatch",
                        files=files,
                        timeout=(5, 120),
                    )
                    if res.status_code == 200:
                        predictions_dict = res.json()
                        preds_series = pd.Series(predictions_dict).astype(float)

                        # Map results back
                        output_df = input_df.copy()
                        output_df['Predicted_Sales'] = preds_series.values

                        st.success("Batch inference complete!")
                        st.write("**Results:**")
                        st.dataframe(output_df)

                        # Download result button
                        csv_buffer = io.StringIO()
                        output_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="superkart_forecasts.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Error: {res.json().get('error', 'Batch inference failed')}")
                except Exception as e:
                    st.error(f"Could not reach the backend service: {e}")
