# imports
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
import functions as fn

# load data
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)

X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])

X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])

def get_X_to_predict():
    X_to_predict = pd.DataFrame({'bedrooms': selected_beds,
                             'bathrooms': selected_baths, 
                             'sqft_lot': selected_lot},
                               index=['House'])
    return X_to_predict

## Title and Markdown subheader
st.title('IMDB Rating Prediction')
## st.image('images/banner.png')
## st.markdown("Data provided by the belt exam.")
                       
st.subheader("Select model and enter text using the sidebar.\n Then check the box below to generate predictions.")
        
s_model = st.sidebar.selectbox('Model',['ml','nlp'])

s_text =st.sidebar.text_input('Insert Text')

# Load model
model = load_model_ml(fpath = FPATHS['models'][s_model])

if st.checkbox("Generate"):
    
    X_to_pred = s_text
    new_pred = fn.get_prediction(model, X_to_pred)
    
    st.markdown(f"> #### {s_model} Model Predicted Rating = {new_pred:,.0f}")
    
else:
    st.empty()
