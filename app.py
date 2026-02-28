import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from microplastic_ftir.preprocessing.pipeline import PreprocessingPipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Microplastic Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    
    with open('preprocessing_config.json') as f:
        pp_config = json.load(f)
    pipeline = PreprocessingPipeline.from_dict(pp_config)
    
    with open('label_decoder.json') as f:
        decoder = json.load(f)
        # Convert keys to int
        decoder = {int(k): v for k, v in decoder.items()}
        
    return model, pipeline, decoder

try:
    model, pipeline, label_decoder = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.title("Settings")
    st.info("This tool identifies microplastic polymers from FTIR spectra using a machine learning model trained on ~6000 samples.")
    
    show_raw = st.checkbox("Show Raw Spectrum", value=True)
    show_proc = st.checkbox("Show Processed Spectrum", value=True)
    
    st.markdown("---")
    st.markdown("Created by **Your Name**")
    st.markdown("Powered by **Streamlit & LightGBM**")

# --- MAIN CONTENT ---
st.title("🔬 Microplastic FTIR Identification")
st.markdown("Upload a CSV or TXT file containing spectral data (Wavenumber, Intensity).")

uploaded_file = st.file_uploader(" ", type=['csv', 'txt'])

if uploaded_file:
    try:
        # 1. READ DATA
        # Try different delimiters
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] < 2:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, sep=r'\s+')
        except:
            st.error("Could not read file. Ensure it is CSV or Space-delimited.")
            st.stop()
            
        wn = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        
        # 2. STANDARDIZE
        target_wn = np.linspace(400, 4000, 1800)
        if wn[0] > wn[-1]: 
            wn = wn[::-1]
            intensity = intensity[::-1]
            
        intensity_interp = np.interp(target_wn, wn, intensity)
        
        # 3. PREPROCESS & PREDICT
        X = intensity_interp.reshape(1, -1)
        X_pp = pipeline.transform(X)
        
        probs = model.predict_proba(X_pp)[0]
        pred_idx = np.argmax(probs)
        pred_name = label_decoder.get(pred_idx, "Unknown")
        confidence = probs[pred_idx]
        
        # --- RESULTS SECTION ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class="prediction-box">
                    <h3 style='margin:0; color:#555;'>Identified Polymer</h3>
                    <h1 style='font-size: 3em; color:#2E86C1; margin:10px 0;'>{pred_name}</h1>
                    <p style='color:#777; font-size:1.1em;'>Confidence: <b>{confidence:.1%}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability Bar Chart
            st.caption("Class Probabilities")
            chart_data = pd.DataFrame({
                "Polymer": [label_decoder[i] for i in range(len(probs))],
                "Probability": probs
            }).sort_values("Probability", ascending=False).head(5)
            
            st.bar_chart(chart_data.set_index("Polymer"))

        with col2:
            # Interactive Spectral Plot (Plotly)
            fig = go.Figure()
            
            if show_raw:
                fig.add_trace(go.Scatter(
                    x=target_wn, y=intensity_interp, 
                    name='Raw Spectrum', 
                    line=dict(color='#888', width=1.5),
                    opacity=0.6
                ))
                
            if show_proc:
                # Scale processed to match raw range for visualization overlay?
                # No, better to use secondary y-axis to fix the "flat line" issue
                fig.add_trace(go.Scatter(
                    x=target_wn, y=X_pp[0], 
                    name='Processed (2nd Axis)', 
                    line=dict(color='#2E86C1', width=2),
                    yaxis='y2'
                ))

            fig.update_layout(
                title="Spectral Analysis",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Raw Intensity",
                yaxis2=dict(
                    title="Processed Intensity",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                xaxis=dict(autorange="reversed"), # Standard FTIR view
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, t=40, b=0),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing spectrum: {e}")
        import traceback
        st.code(traceback.format_exc())
