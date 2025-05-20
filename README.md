# 🚗 Nigerian Traffic Crash Predictor (2020-2024)

AI-powered analysis and prediction of traffic crash patterns using Cohere's LLM and linear regression.

## Features
- 📊 Exploratory data analysis
- 🧠 Linear Regression modeling (MAE, MSE, R²)
- 💬 LLM-powered Q&A about the dataset
- 🚀 Streamlit interactive interface

## Setup
1. Clone repo:
   ```bash
   git clone https://github.com/yourusername/traffic-crash-predictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your [Cohere API key](https://dashboard.cohere.com/api-keys) in `.env`:
   ```env
   COHERE_API_KEY="your_key_here"
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## Screenshot
![App Preview](screenshot.png)

## Dataset
Sample dataset included in `/data`. Replace with your own CSV matching:
- `Total_Crashes`, `Num_Injured`, `Num_Killed`, etc. columns
