import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page settings
st.set_page_config(page_title="Machine Failure Predictor", layout="wide")

# UI Style
st.markdown(
    '''
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #e8f5e9, #fffde7);
        animation: fadeIn 2s ease-in;
    }
    h1 {
        color: #1b5e20;
        text-align: center;
        font-size: 36px;
        margin-top: 10px;
    }
    .intro {
        font-size: 18px;
        text-align: center;
        font-weight: 400;
        color: #333;
        margin-bottom: 20px;
    }
    @media only screen and (max-width: 768px) {
        h1 { font-size: 26px; }
        .intro { font-size: 16px; }
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs(["📊 Predict", "📘 About"])

# ----------- Predict Tab ----------
with tab1:
    st.markdown("<h1>🚀 Machine Failure Predictor</h1>", unsafe_allow_html=True)
    st.markdown('<div class="intro">Upload machine sensor data to predict failure risk using AI.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload your sensor data CSV", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(data.head(), use_container_width=True)

            with st.spinner("🔄 Processing..."):
                data.columns = data.columns.str.strip()
                features = data.loc[:, ~data.columns.str.lower().isin(['fail'])]

                scaled = scaler.transform(features)
                preds = model.predict(scaled)
                probs = model.predict_proba(scaled)[:, 1]

                data['Predicted Failure'] = preds
                data['Failure Risk (%)'] = (probs * 100).round(2)

            st.success("✅ Prediction complete!")

            st.subheader("📊 Results")
            st.dataframe(data[['Predicted Failure', 'Failure Risk (%)']], use_container_width=True)

            st.subheader("📈 Risk Trend")
            st.line_chart(data['Failure Risk (%)'])

            # Download predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "predicted_results.csv", "text/csv")

            high_risk = data[data['Failure Risk (%)'] > 80]
            if not high_risk.empty:
                st.warning(f"⚠️ {len(high_risk)} machines are at HIGH risk of failure!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📁 Please upload a CSV file to begin.")

# ----------- About Tab ----------
with tab2:
    st.markdown("<h1>📘 About This Capstone</h1>", unsafe_allow_html=True)
    st.markdown('''
    <div class="intro">
    This capstone project uses a machine learning model to predict machine failure from real-time sensor readings.<br><br>
    🔍 **Features Used**:
    - Footfall, Temp Mode, AQ, USS, CS, VOC, RP, IP, Temperature<br>
    - Trained on historical labeled data<br><br>
    🎯 **Goal**: Enable industries to avoid unplanned downtime with proactive maintenance.<br><br>
    💡 Built using **Python, Streamlit, scikit-learn**, and deployed for interviews and production-ready presentation.
    </div>
    ''', unsafe_allow_html=True)
