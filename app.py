import streamlit as st
import pandas as pd
import numpy as np
import cohere
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Traffic Crash Predictor with LLM", layout="centered")

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client if API key is present
if cohere_api_key:
    co = cohere.Client(cohere_api_key)
else:
    co = None
    st.warning("⚠️ Cohere API key not found. Add it to your `.env` file as COHERE_API_KEY.")

# Title
st.title("🚗 Nigerian Traffic Crash Predictor (2020–2024)")

# Load dataset
try:
    df = pd.read_csv("/Users/macbook/Desktop/traffic_llm_app/nigeria_traffic_data.csv")
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    df = None

# Proceed if data loaded
if df is not None:
    st.markdown("---")

    # Feature selection
    features = ['Num_Injured', 'Num_Killed', 'Total_Vehicles_Involved', 'SPV', 'DAD', 'FTQ', 'Other_Factors']
    target = 'Total_Crashes'

    try:
        X = df[features]
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Model Evaluation
        st.subheader("📊 Model Evaluation (Linear Regression)")
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**MSE**: {mse:.2f}")
        st.write(f"**RMSE**: {rmse:.2f}")
        st.write(f"**R² Score**: {r2:.5f}")

        # Prediction Input
        st.markdown("---")
        st.header("🧠 Predict Total Crashes")

        Num_Injured = st.number_input("Number Injured", min_value=0)
        Num_Killed = st.number_input("Number Killed", min_value=0)
        Total_Vehicles = st.number_input("Total Vehicles Involved", min_value=0)
        SPV = st.number_input("SPV", min_value=0)
        DAD = st.number_input("DAD", min_value=0)
        FTQ = st.number_input("FTQ", min_value=0)
        Other_Factors = st.number_input("Other Factors", min_value=0)

        if st.button("🚀 Predict Total Crashes"):
            input_data = [[Num_Injured, Num_Killed, Total_Vehicles, SPV, DAD, FTQ, Other_Factors]]
            prediction = model.predict(input_data)
            st.success(f"📌 Predicted Total Crashes: **{prediction[0]:.0f}**")
    except Exception as e:
        st.error(f"❌ Error during modeling: {e}")

    
st.markdown("---")
st.subheader("💬 Ask About the Dataset (LLM Powered)")

question = st.text_input("Ask a question about the dataset")

if st.button("Ask LLM"):
    if co:
        try:
            sample_data = df.to_csv(index=False)[:4000]  # Truncate CSV
            response = co.chat(
                model="command-r",
                message=question,
                documents=[{"text": sample_data}]  # ✅ Fixed: wrapped in dict
            )
            answer = response.text.strip()
            st.success(f"💡 LLM Answer: {answer}")
        except Exception as e:
            st.error(f"❌ LLM error: {e}")
    else:
        st.error("Cohere client not initialized. Make sure your API key is correct.")