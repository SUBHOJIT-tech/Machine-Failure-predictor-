import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Smart Failure Predictor", page_icon="⚙️", layout="wide")

# Custom CSS for background + animations
st.markdown(
    """
    <style>
    body, html {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f0f4c3, #e1f5fe);
        animation: backgroundFade 15s ease-in-out infinite alternate;
    }
    @keyframes backgroundFade {
        0% { background-color: #f0f4c3; }
        100% { background-color: #e1f5fe; }
    }
    h1 {
        font-size: 42px;
        font-weight: bold;
        color: #1a237e;
        text-align: center;
        margin-bottom: 5px;
    }
    h2 {
        text-align: center;
        color: #424242;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #757575;
        margin-top: 30px;
    }
    @media screen and (max-width: 768px) {
        h1 { font-size: 30px; }
        h2 { font-size: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs(["📊 Predict", "ℹ️ About"])

# --- Prediction Tab ---
with tab1:
    st.markdown("<h1>⚙️ Smart Machine Failure Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Upload your sensor data to see failure risk predictions using AI.</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.strip()

            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(data.head(), use_container_width=True)

            with st.spinner("🔄 Analyzing sensor data..."):
                # Remove "Fail" column safely (case-insensitive)
                features = data.loc[:, ~data.columns.str.lower().isin(['fail'])]

                # Predict
                scaled = scaler.transform(features)
                preds = model.predict(scaled)
                probs = model.predict_proba(scaled)[:, 1]

                # Add results
                data['Predicted Failure'] = preds
                data['Failure Risk (%)'] = (probs * 100).round(2)

            st.success("✅ Analysis complete!")

            st.subheader("📊 Prediction Output")
            st.dataframe(data[['Predicted Failure', 'Failure Risk (%)']], use_container_width=True)

            st.subheader("📈 Failure Risk Chart")
            st.line_chart(data['Failure Risk (%)'])

            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result CSV", csv, "machine_failure_predictions.csv", "text/csv")

            # High risk warning
            high_risk = data[data['Failure Risk (%)'] > 80]
            if not high_risk.empty:
                st.warning(f"⚠️ {len(high_risk)} machines are at HIGH risk of failure!")

        except Exception as e:
            st.error(f"❌ An error occurred while processing your file: {e}")
    else:
        st.info("📁 Please upload a CSV file to begin.")

    st.markdown("<div class='footer'>Built with ❤️ by Subhojit | AI-powered reliability</div>", unsafe_allow_html=True)

# --- About Tab ---
with tab2:
    st.markdown("<h1>📘 About This Project</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; font-size: 17px;'>
        This capstone project uses machine learning to intelligently predict potential failures in machines
        based on real-time sensor readings. <br><br>
        ✅ Model Trained On: Historical machine sensor data<br>
        ✅ Tools Used: Python, scikit-learn, Streamlit<br>
        ✅ Deployed With: Streamlit Cloud<br><br>
        <strong>Presented by:</strong> Subhojit | Data Science Capstone | 2025
        </div>
        """,
        unsafe_allow_html=True
    )

