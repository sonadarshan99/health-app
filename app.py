import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="HealthCalc AI Pro", page_icon="💰", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- Step 1: Data Engine ---
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    plot_df = df.copy()
    
    # Encoding for the AI Model
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])
    
    return df, plot_df

try:
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
    children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
    smoker = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
    region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # --- Step 3: AI Training & Prediction ---
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = r2_score(y_test, model.predict(X_test))

    # Prediction Logic
    sex_val = 0 if sex == "Female" else 1
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_map[region]]])
    raw_pred = model.predict(input_data)[0]
    final_pred = raw_pred * rate

    # --- Step 4: Main UI ---
    st.title("🏥 Healthcare AI Cost Predictor")
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])

    with col1:
        st.subheader("Cost Estimate")
        st.metric(label=f"Annual Charges ({currency})", value=f"{symbol}{final_pred:,.0f}")
        st.info(f"Model Confidence Score: **{accuracy:.2%}**")

    with col2:
        st.subheader("Health Status")
        if bmi < 18.5: status, color = "Underweight", "blue"
        elif 18.5 <= bmi <= 24.9: status, color = "Normal", "green"
        elif 25 <= bmi <= 29.9: status, color = "Overweight", "orange"
        else: status, color = "Obese", "red"
        st.markdown(f"BMI Status: **:{color}[{status}]**")
        
    with col3:
        st.subheader("💡 Pro-Tip")
        if smoker == "Yes":
            st.warning("Quitting smoking could reduce your projected annual cost by nearly 60%.")
        elif bmi > 25:
            st.success("Bringing your BMI into the 'Normal' range could lower costs by 15-20%.")
        else:
            st.success("You're in a low-risk category! Keep maintaining your healthy lifestyle.")

    # --- Step 5: Visualizations ---
    st.divider()
    t1, t2 = st.tabs(["📊 Market Trends", "🧠 AI Logic"])

    with t1:
        plot_df['charges_conv'] = plot_df['charges'] * rate
        fig = px.scatter(plot_df, x="bmi", y="charges_conv", color="smoker",
                         title=f"Cost Analysis in {currency}",
                         labels={"charges_conv": f"Charges ({symbol})", "bmi": "BMI"},
                         color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"})
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        feat_imp = pd.DataFrame({'Factor': X.columns, 'Weight': model.feature_importances_}).sort_values('Weight')
        fig_imp = px.bar(feat_imp, x='Weight', y='Factor', orientation='h', 
                         title="Which factors affect your price most?",
                         color='Weight', color_continuous_scale='GnBu')
        st.plotly_chart(fig_imp, use_container_width=True)

    # Export Feature
    report = f"Patient Report\nAge: {age}\nBMI: {bmi}\nSmoker: {smoker}\nEstimated Cost: {symbol}{final_pred:,.0f}"
    st.sidebar.download_button("📥 Download Report", report, file_name="health_report.txt")

except Exception as e:
    st.error(f"Please ensure 'insurance.csv' is in your GitHub repo. Error: {e}")
