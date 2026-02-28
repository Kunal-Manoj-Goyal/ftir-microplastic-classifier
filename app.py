import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from io import BytesIO
from microplastic_ftir.preprocessing.pipeline import PreprocessingPipeline

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SpectraMind | Microplastic ID",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"  # Cleaner look
)

# --- 2. CUSTOM CSS (Modern UI) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header Styles */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #4facfe;
        margin-bottom: 0px;
    }
    h3 {
        font-weight: 400;
        color: #A0AEC0;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, #1e212b, #242731);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .polymer-name {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .confidence-badge {
        background-color: #2E86C1;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Custom File Uploader Style */
    .stFileUploader {
        border: 2px dashed #4facfe;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Button Style */
    .stButton>button {
        background-color: #2E3B55;
        color: white;
        border: 1px solid #4facfe;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #4facfe;
        border-color: #00f2fe;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        with open('preprocessing_config.json') as f:
            pp_config = json.load(f)
        pipeline = PreprocessingPipeline.from_dict(pp_config)
        with open('label_decoder.json') as f:
            decoder = json.load(f)
            decoder = {int(k): v for k, v in decoder.items()}
        return model, pipeline, decoder
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'model.pkl' and configs are in the root directory.")
        st.stop()

def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

# --- 4. APP LOGIC ---
model, pipeline, label_decoder = load_resources()

# -- Hero Section --
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=80)
with col_title:
    st.title("SpectraMind")
    st.caption("Advanced AI for Microplastic Identification")

st.markdown("---")

# -- Layout --
col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.markdown("### 1. Upload Spectrum")
    st.info("Supported formats: CSV, TXT (2 columns: Wavenumber, Intensity)")
    uploaded_file = st.file_uploader("Drop your spectral file here", type=['csv', 'txt'], label_visibility="collapsed")

if uploaded_file:
    try:
        # Data Loading logic (Robust)
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] < 2:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, sep=r'\s+')
        except:
            st.error("Error reading file format.")
            st.stop()

        wn_raw = df.iloc[:, 0].values
        int_raw = df.iloc[:, 1].values

        # Standardize (Interpolate to 1800 pts)
        target_wn = np.linspace(400, 4000, 1800)
        # Handle descending/ascending raw data
        if wn_raw[0] > wn_raw[-1]:
            wn_sorted = wn_raw[::-1]
            int_sorted = int_raw[::-1]
        else:
            wn_sorted = wn_raw
            int_sorted = int_raw
            
        intensity_interp = np.interp(target_wn, wn_sorted, int_sorted)

        # Preprocess
        X = intensity_interp.reshape(1, -1)
        X_pp = pipeline.transform(X) # This is the "Processed" data (Derivative/SNV)

        # Predict
        probs = model.predict_proba(X_pp)[0]
        pred_idx = np.argmax(probs)
        pred_name = label_decoder.get(pred_idx, "Unknown")
        confidence = probs[pred_idx]

        # --- RESULTS COLUMN ---
        with col_result:
            st.markdown("### 2. Analysis Result")
            
            # The "Card"
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin-bottom:0;">Identified Polymer</h3>
                    <div class="polymer-name">{pred_name}</div>
                    <span class="confidence-badge">Confidence: {confidence:.1%}</span>
                </div>
            """, unsafe_allow_html=True)

            # Probabilities Chart (Plotly for interactivity)
            top_n = 5
            top_indices = np.argsort(probs)[::-1][:top_n]
            top_probs = probs[top_indices]
            top_names = [label_decoder.get(i, f"Class {i}") for i in top_indices]

            fig_bar = go.Figure(go.Bar(
                x=top_probs,
                y=top_names,
                orientation='h',
                marker=dict(color=top_probs, colorscale='Blues')
            ))
            fig_bar.update_layout(
                title="Top Matches",
                xaxis_title="Probability",
                yaxis=dict(autorange="reversed"),
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.stop()

# --- FULL WIDTH PLOT SECTION ---
if uploaded_file:
    st.markdown("---")
    st.markdown("### 3. Spectral Visualization")
    
    # Create clean dataframe for export
    export_df = pd.DataFrame({
        "Wavenumber": target_wn,
        "Processed_Intensity": X_pp[0]
    })

    col_chart, col_download = st.columns([5, 1])
    
    with col_download:
        st.write("") # Spacer
        st.markdown("**Export Data**")
        st.download_button(
            label="Download .CSV",
            data=to_csv(export_df),
            file_name=f"{pred_name}_processed_spectrum.csv",
            mime="text/csv",
            help="Download the preprocessed spectral data used for prediction."
        )

    with col_chart:
        # Main Spectral Plot
        fig_spec = go.Figure()
        
        # Plot only the PROCESSED data (clean signal)
        fig_spec.add_trace(go.Scatter(
            x=target_wn, 
            y=X_pp[0], 
            mode='lines',
            name='Processed Spectrum',
            line=dict(color='#00f2fe', width=2),
            fill='tozeroy',  # Nice area fill effect
            fillcolor='rgba(0, 242, 254, 0.1)'
        ))

        fig_spec.update_layout(
            title="Processed Signal (Derivative/SNV)",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Normalized Intensity",
            xaxis=dict(autorange="reversed", gridcolor='#333'), # FTIR standard reverse X
            yaxis=dict(gridcolor='#333'),
            height=500,
            hovermode="x unified",
            paper_bgcolor='#0E1117',
            plot_bgcolor='#161b24',
            font=dict(color='#FAFAFA'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_spec, use_container_width=True)
