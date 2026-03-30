import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import io

# --- Page Config ---
st.set_page_config(page_title="HealthCalc AI", page_icon="🏥", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Title Section ---
st.title("🏥 Healthcare Cost Prediction & Analytics")
st.markdown("Predict annual medical charges and explore the underlying data trends using Machine Learning.")

# --- Step 1: Load & Prepare Data ---
@st.cache_data
def load_and_process_data():
    # Ensure insurance.csv is in the same folder
    df = pd.read_csv('insurance.csv')
    
    # We keep a copy of the original for plotting
    plot_df = df.copy()
    
    # Encoding categorical variables for the Model
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])
    
    return df, plot_df

try:
    df, plot_df = load_and_process_data()
    
    # --- Step 2: Model Training (Random Forest) ---
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    # --- Sidebar: User Inputs ---
    st.sidebar.header("📝 Patient Information")
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
    children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
    region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # Map inputs back to encoded values for prediction
    sex_val = 0 if sex == "Female" else 1
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region_val = region_map[region]

    # --- Prediction Result ---
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Results")
        st.metric(label="Estimated Annual Cost", value=f"${prediction:,.2f}")
        st.write(f"**Model Confidence:** {accuracy:.2%}")
        
        # --- Download Feature ---
        report_text = f"""
        HEALTHCARE COST REPORT
        ----------------------
        Age: {age}
        Gender: {sex}
        BMI: {bmi}
        Children: {children}
        Smoker: {smoker}
        Region: {region}
        
        ESTIMATED COST: ${prediction:,.2f}
        Model Accuracy: {accuracy:.2%}
        """
        st.download_button(
            label="📄 Download Prediction Report",
            data=report_text,
            file_name=f"health_report_{age}.txt",
            mime="text/plain"
        )

    with col2:
        # Feature Importance Chart
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance')
        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                         title="Key Factors Influencing Your Price",
                         color='Importance', color_continuous_scale='Blues')
        fig_imp.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- Data Visualizations Section ---
    st.divider()
    st.subheader("📊 Population Data Trends")
    
    tab1, tab2 = st.tabs(["BMI & Smoking Impact", "Regional Distribution"])
    
    with tab1:
        fig_bmi = px.scatter(plot_df, x="bmi", y="charges", color="smoker",
                             title="How BMI and Smoking drive Costs",
                             labels={"bmi": "BMI", "charges": "Medical Charges ($)", "smoker": "Smoker"},
                             color_discrete_map={"yes": "#ef553b", "no": "#636efa"})
        st.plotly_chart(fig_bmi, use_container_width=True)
    
    with tab2:
        reg_avg = plot_df.groupby('region')['charges'].mean().reset_index()
        fig_reg = px.pie(reg_avg, values='charges', names='region', title="Average Cost Distribution by Region",
                         hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_reg, use_container_width=True)

    st.divider()
    with st.expander("See Raw Dataset Sample"):
        st.write(plot_df.head(10))

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Ensure 'insurance.csv' is in the same directory as this app.")