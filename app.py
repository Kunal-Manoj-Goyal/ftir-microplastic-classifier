
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from microplastic_ftir.preprocessing.pipeline import PreprocessingPipeline

# Load Config
with open('preprocessing_config.json') as f:
    pp_config = json.load(f)

# Load Model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
pipeline = PreprocessingPipeline.from_dict(pp_config)

st.title("Microplastic Classifier 🔬")
st.write(f"Using Best Model: **lightgbm**")

uploaded_file = st.file_uploader("Upload FTIR Spectrum (CSV/TXT)", type=['csv', 'txt'])

if uploaded_file:
    # 1. Read Data (Simple Loader Logic)
    try:
        df = pd.read_csv(uploaded_file, header=None)
        # Assume col 0 = wavenumber, col 1 = intensity
        wn = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        
        # 2. Interpolate to 1800 points (Standardization)
        target_wn = np.linspace(400, 4000, 1800)
        # Sort if needed
        if wn[0] > wn[-1]: 
            wn = wn[::-1]
            intensity = intensity[::-1]
            
        intensity_interp = np.interp(target_wn, wn, intensity)
        
        # 3. Preprocess
        X = intensity_interp.reshape(1, -1)
        X_pp = pipeline.transform(X)
        
        # 4. Predict
        # Handle Wrapper vs Raw Model
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_pp)[0]
            pred = model.predict(X_pp)[0]
        elif hasattr(model, 'model'): # TraditionalWrapper
            probs = model.predict_proba(X_pp)[0]
            pred = model.predict(X_pp)[0]
            
        # 5. Display
        st.success(f"Prediction: **{pred}**")
        
        # Plot
        fig, ax = plt.subplots()
        ax.plot(target_wn, X[0], label='Raw')
        ax.plot(target_wn, X_pp[0], label='Processed', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        
        st.bar_chart(probs)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
