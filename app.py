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
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (Premium Modern UI) ---
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.03) 0%, transparent 50%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-top: 10px;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        position: relative;
        z-index: 1;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .section-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }
    
    /* Upload Card */
    .upload-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    
    .upload-zone {
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .upload-text {
        color: #a0aec0;
        font-size: 0.95rem;
    }
    
    .upload-formats {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .result-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .polymer-name {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 15px 0;
        line-height: 1.2;
    }
    
    .confidence-container {
        margin-top: 25px;
    }
    
    .confidence-bar-bg {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        height: 12px;
        overflow: hidden;
        margin: 12px 0;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        transition: width 1s ease-out;
    }
    
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .confidence-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 25px;
    }
    
    .stat-item {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #a0aec0;
        margin-top: 4px;
    }
    
    /* Chart Card */
    .chart-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin-top: 10px;
    }
    
    /* Download Button Custom */
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
    }
    
    /* Streamlit Overrides */
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stFileUploader label {
        display: none;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 15px 20px;
        margin: 15px 0;
    }
    
    .info-box p {
        color: #a0aec0;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 40px 0;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    /* Probability Pills */
    .prob-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 15px;
    }
    
    .prob-item {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .prob-name {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
        min-width: 100px;
    }
    
    .prob-bar-container {
        flex: 1;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    
    .prob-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
    }
    
    .prob-value {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        min-width: 50px;
        text-align: right;
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
st.markdown("""
    <div class="hero-container">
        <span class="hero-badge">🔬 AI-Powered Analysis</span>
        <h1 class="hero-title">SpectraMind</h1>
        <p class="hero-subtitle">Advanced machine learning for microplastic polymer identification from FTIR spectral data</p>
    </div>
""", unsafe_allow_html=True)

# -- Main Layout --
col_upload, col_spacer, col_result = st.columns([1, 0.1, 1.5])

with col_upload:
    st.markdown("""
        <div class="section-header">
            <div class="section-number">1</div>
            <h3 class="section-title">Upload Spectrum</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="upload-card">
            <div class="upload-zone">
                <div class="upload-icon">📊</div>
                <p class="upload-text">Drag and drop your spectral file here<br>or click to browse</p>
                <span class="upload-formats">CSV • TXT</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("upload", type=['csv', 'txt'], label_visibility="collapsed")
    
    st.markdown("""
        <div class="info-box">
            <p>📋 <strong>Format:</strong> Two columns (Wavenumber, Intensity)<br>
            📏 <strong>Range:</strong> 400-4000 cm⁻¹ recommended</p>
        </div>
    """, unsafe_allow_html=True)

# Initialize variables
pred_name = None
target_wn = None
X_pp = None

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
        X_pp = pipeline.transform(X)

        # Predict
        probs = model.predict_proba(X_pp)[0]
        pred_idx = np.argmax(probs)
        pred_name = label_decoder.get(pred_idx, "Unknown")
        confidence = probs[pred_idx]

        # --- RESULTS COLUMN ---
        with col_result:
            st.markdown("""
                <div class="section-header">
                    <div class="section-number">2</div>
                    <h3 class="section-title">Analysis Result</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Result Card
            st.markdown(f"""
                <div class="result-card animate-in">
                    <p class="result-label">Identified Polymer</p>
                    <h2 class="polymer-name">{pred_name}</h2>
                    
                    <div class="confidence-container">
                        <span class="confidence-value">{confidence:.1%}</span>
                        <span class="confidence-label"> confidence</span>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {confidence*100}%;"></div>
                        </div>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{len(wn_raw)}</div>
                            <div class="stat-label">Data Points</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{wn_raw.min():.0f}-{wn_raw.max():.0f}</div>
                            <div class="stat-label">Range (cm⁻¹)</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{len(label_decoder)}</div>
                            <div class="stat-label">Polymers DB</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Top Matches Section
            st.markdown("<br>", unsafe_allow_html=True)
            
            top_n = 5
            top_indices = np.argsort(probs)[::-1][:top_n]
            top_probs = probs[top_indices]
            top_names = [label_decoder.get(i, f"Class {i}") for i in top_indices]

            # Plotly Chart with enhanced styling
            colors = ['#667eea', '#764ba2', '#f093fb', '#a855f7', '#6366f1']
            
            fig_bar = go.Figure(go.Bar(
                x=top_probs,
                y=top_names,
                orientation='h',
                marker=dict(
                    color=colors[:len(top_probs)],
                    line=dict(width=0)
                ),
                text=[f'{p:.1%}' for p in top_probs],
                textposition='inside',
                textfont=dict(color='white', size=12, family='Inter'),
                hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
            ))
            
            fig_bar.update_layout(
                title=dict(
                    text="<b>Top Polymer Matches</b>",
                    font=dict(size=16, color='#ffffff', family='Inter'),
                    x=0
                ),
                xaxis=dict(
                    title="Probability",
                    titlefont=dict(color='#a0aec0', size=12),
                    tickfont=dict(color='#a0aec0', size=10),
                    gridcolor='rgba(102, 126, 234, 0.1)',
                    tickformat='.0%',
                    range=[0, 1]
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color='#ffffff', size=12, family='Inter')
                ),
                height=280,
                margin=dict(l=0, r=20, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                bargap=0.4,
                showlegend=False
            )
            
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Processing Error: {e}")
        st.stop()

# --- FULL WIDTH PLOT SECTION ---
if uploaded_file and pred_name and target_wn is not None and X_pp is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col_header, col_download = st.columns([4, 1])
    
    with col_header:
        st.markdown("""
            <div class="section-header">
                <div class="section-number">3</div>
                <h3 class="section-title">Spectral Visualization</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col_download:
        export_df = pd.DataFrame({
            "Wavenumber": target_wn,
            "Processed_Intensity": X_pp[0]
        })
        st.download_button(
            label="📥 Export CSV",
            data=to_csv(export_df),
            file_name=f"{pred_name}_processed_spectrum.csv",
            mime="text/csv",
            help="Download the preprocessed spectral data"
        )

    # Main Spectral Plot
    fig_spec = go.Figure()
    
    fig_spec.add_trace(go.Scatter(
        x=target_wn, 
        y=X_pp[0], 
        mode='lines',
        name='Processed Spectrum',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.15)',
        hovertemplate='<b>Wavenumber:</b> %{x:.1f} cm⁻¹<br><b>Intensity:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add gradient effect with additional traces
    fig_spec.add_trace(go.Scatter(
        x=target_wn,
        y=X_pp[0],
        mode='lines',
        line=dict(color='#764ba2', width=1),
        opacity=0.5,
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_spec.update_layout(
        title=dict(
            text="<b>Processed FTIR Spectrum</b> <span style='color:#a0aec0;font-weight:400;font-size:14px;'>| Derivative/SNV Normalized</span>",
            font=dict(size=18, color='#ffffff', family='Inter'),
            x=0
        ),
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            titlefont=dict(color='#a0aec0', size=13, family='Inter'),
            tickfont=dict(color='#a0aec0', size=11),
            autorange="reversed",
            gridcolor='rgba(102, 126, 234, 0.1)',
            linecolor='rgba(102, 126, 234, 0.3)',
            zeroline=False,
            showspikes=True,
            spikecolor='#667eea',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title="Normalized Intensity",
            titlefont=dict(color='#a0aec0', size=13, family='Inter'),
            tickfont=dict(color='#a0aec0', size=11),
            gridcolor='rgba(102, 126, 234, 0.1)',
            linecolor='rgba(102, 126, 234, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(102, 126, 234, 0.3)',
            zerolinewidth=1
        ),
        height=500,
        hovermode="x unified",
        paper_bgcolor='rgba(30, 30, 47, 0.8)',
        plot_bgcolor='rgba(20, 20, 35, 0.9)',
        font=dict(family='Inter'),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#a0aec0')
        )
    )
    
    # Add range slider
    fig_spec.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.05,
            bgcolor='rgba(30, 30, 47, 0.5)',
            bordercolor='rgba(102, 126, 234, 0.3)'
        )
    )
    
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_spec, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0; color: #4a5568;">
        <p style="font-size: 0.85rem;">
            Built with 💜 using Streamlit & Machine Learning<br>
            <span style="color: #667eea;">SpectraMind</span> © 2024
        </p>
    </div>
""", unsafe_allow_html=True)
