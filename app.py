import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="HealthCalc AI Pro", page_icon="🏥", layout="wide")

# --- High-Visibility Custom CSS ---
st.markdown("""
    <style>
    /* Force main background to light gray */
    .stApp {
        background-color: #f4f7f6;
    }
    /* Style the metric value to be Deep Blue and Bold */
    [data-testid="stMetricValue"] {
        color: #004085 !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }
    /* Style the metric label to be Dark Gray */
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    /* Create white 'Card' containers for each column */
    div[data-testid="column"] {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 10px;
    }
    /* Ensure subheaders are visible */
    h3 {
        color: #2c3e50 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Step 1: Data Engine ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('insurance.csv')
        plot_df = df.copy()
        
        # Encoding for the AI Model
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
        df['smoker'] = le.fit_transform(df['smoker'])
        df['region'] = le.fit_transform(df['region'])
        
        return df, plot_df
    except FileNotFoundError:
        st.error("⚠️ File 'insurance.csv' not found. Please upload it to your GitHub repo.")
        st.stop()

df, plot_df = load_data()

# --- Step 2: Sidebar Settings & Inputs ---
st.sidebar.header("⚙️ App Settings")
currency = st.sidebar.radio("Select Currency", ["Rupees (₹)", "US Dollars ($)"])
rate = 83.0 if currency == "Rupees (₹)" else 1.0
symbol = "₹" if currency == "Rupees (₹)" else "$"

st.sidebar.divider()
st.sidebar.header("📝 Patient Details")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.selectbox("Children/Dependents", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# --- Step 3: AI Training & Prediction ---
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest is best for this specific dataset
model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# Prediction Calculation
sex_val = 0 if sex == "Female" else 1
smoker_val = 1 if smoker == "Yes" else 0
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

input_features = np.array([[age, sex_val, bmi, children, smoker_val, region_map[region]]])
raw_pred = model.predict(input_features)[0]
final_pred = raw_pred * rate

# --- Step 4: Main UI Layout ---
st.title("🏥 Health Insurance Cost Predictor")
st.markdown("Adjust the sidebar values to see real-time cost estimations.")

col1, col2, col3 = st.columns([1.5, 1, 1.5])

with col1:
    st.subheader("Cost Estimate")
    # This metric is now high-contrast Blue on White
    st.metric(label=f"Annual Charges ({currency})", value=f"{symbol}{final_pred:,.0f}")
    st.info(f"Model Accuracy (R²): **{accuracy:.2%}**")

with col2:
    st.subheader("Health Status")
    if bmi < 18.5: status, color = "Underweight", "blue"
    elif 18.5 <= bmi <= 24.9: status, color = "Normal", "green"
    elif 25 <= bmi <= 29.9: status, color = "Overweight", "orange"
    else: status, color = "Obese", "red"
    st.markdown(f"Your BMI Category: **:{color}[{status}]**")
    
with col3:
    st.subheader("💡 Analysis")
    if smoker == "Yes":
        st.warning("⚠️ High Impact: Smoking is the #1 driver for your increased cost.")
    elif bmi > 30:
        st.warning("⚠️ Weight Factor: Being in the Obese range increases health premiums.")
    else:
        st.success("✅ Good Standing: You have a healthy risk profile.")

# --- Step 5: Visualizations ---
st.divider()
t1, t2 = st.tabs(["📊 Market Data Trends", "🧠 AI Feature Weights"])

with t1:
    plot_df['charges_conv'] = plot_df['charges'] * rate
    fig = px.scatter(plot_df, x="bmi", y="charges_conv", color="smoker",
                     title=f"Healthcare Cost Distribution ({currency})",
                     labels={"charges_conv": f"Charges ({symbol})", "bmi": "BMI"},
                     color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    feat_imp = pd.DataFrame({'Factor': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig_imp = px.bar(feat_imp, x='Importance', y='Factor', orientation='h', 
                     title="What factors weigh most in the AI's decision?",
                     color='Importance', color_continuous_scale='Blues',
                     template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

# Export Feature in Sidebar
report = f"""HEALTH INSURANCE COST REPORT
--------------------------------
Age: {age}
Gender: {sex}
BMI: {bmi} ({status})
Smoker: {smoker}
Region: {region}
--------------------------------
Estimated Annual Cost: {symbol}{final_pred:,.0f}
Model Confidence: {accuracy:.2%}
"""
st.sidebar.download_button("📥 Download My Report", report, file_name=f"health_report_{age}.txt")
