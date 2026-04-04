import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 1. Helper Function to generate/load the Scaler
@st.cache_resource
def prepare_assets():
    # Load the new model
    model = joblib.load('best_gradient_boosting_model.pkl')
    
    # Load CSV to recreate the scaler (since model was trained on scaled data)
    df = pd.read_csv('Life Expectancy Data.csv')
    
    # Preprocessing to match the training pipeline in the notebook
    df = df.drop_duplicates(subset=['Country', 'Year'])
    df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})
    df_clean = df.drop(columns=['Country']).rename(columns={'Life expectancy ': 'Life_expectancy'})
    
    # Simple Imputation (as done in the notebook)
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # Prepare Features (X)
    X = df_clean.drop(columns=['Life_expectancy'])
    
    # Fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    return model, scaler, X.columns.tolist()

# Initialize Model and Scaler
try:
    model, scaler, feature_columns = prepare_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure 'best_gradient_boosting_model.pkl' and 'Life Expectancy Data.csv' are in the folder.")
    st.stop()

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")
st.title("🌍 Life Expectancy Prediction App")
st.write("Enter health and economic indicators to predict the life expectancy of a population.")

# 2. Input Fields Layout
st.subheader("Input Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", 2000, 2025, 2015)
    status = st.selectbox("Status", ["Developed", "Developing"])
    adult_mortality = st.number_input("Adult Mortality (per 1000)", 1, 1000, 263)
    infant_deaths = st.number_input("Infant Deaths (per 1000)", 0, 2000, 62)
    alcohol = st.number_input("Alcohol Consumption (litres)", 0.0, 20.0, 0.01)
    perc_exp = st.number_input("Percentage Expenditure (%)", 0.0, 20000.0, 71.2)
    hep_b = st.number_input("Hepatitis B Coverage (%)", 0, 100, 65)

with col2:
    measles = st.number_input("Measles (reported cases)", 0, 300000, 1154)
    bmi = st.number_input("Average BMI", 1.0, 100.0, 19.1)
    under_five = st.number_input("Under-five Deaths", 0, 3000, 83)
    polio = st.number_input("Polio Coverage (%)", 0, 100, 6)
    total_exp = st.number_input("Total Govt Expenditure (%)", 0.0, 20.0, 8.1)
    diphtheria = st.number_input("Diphtheria Coverage (%)", 0, 100, 65)
    hiv_aids = st.number_input("HIV/AIDS Prevalence (%)", 0.0, 60.0, 0.1)

with col3:
    gdp = st.number_input("GDP (USD per capita)", 0.0, 120000.0, 584.2)
    population = st.number_input("Population", 0.0, 2e9, 3.3e7)
    thin_1_19 = st.number_input("Thinness 1-19 years (%)", 0.0, 30.0, 17.2)
    thin_5_9 = st.number_input("Thinness 5-9 years (%)", 0.0, 30.0, 17.3)
    income_comp = st.number_input("Income Composition of Resources", 0.0, 1.0, 0.47)
    schooling = st.number_input("Schooling (years)", 0.0, 25.0, 10.1)

# 3. Prediction Logic
if st.button("Predict Life Expectancy", type="primary"):
    # Preprocess status to match training (Developed=1, Developing=0)
    status_numeric = 1 if status == "Developed" else 0
    
    # Create feature array in exact order: 
    # ['Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 
    #  'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 
    #  'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 
    #  ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 
    #  ' thinness 5-9 years', 'Income composition of resources', 'Schooling']
    
    raw_features = np.array([[
        year, status_numeric, adult_mortality, infant_deaths, alcohol,
        perc_exp, hep_b, measles, bmi, under_five, 
        polio, total_exp, diphtheria, hiv_aids, gdp, 
        population, thin_1_19, thin_5_9, income_comp, schooling
    ]])
    
    try:
        # 4. Apply the same scaling used in training
        scaled_features = scaler.transform(raw_features)
        
        # 5. Predict using the Gradient Boosting model
        prediction = model.predict(scaled_features)
        
        # Display Result
        st.success(f"### Predicted Life Expectancy: {prediction[0]:.1f} years")
        
        # Progress bar for visual appeal
        st.progress(min(max(prediction[0]/100, 0.0), 1.0))
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")