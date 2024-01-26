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
import exam_functions as fn

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
                       
st.sidebar.subheader('House Features')
                       
s_model = st.sidebar.selectbox('Model',['ML','NLP'])

selected_baths = st.sidebar.slider('Number of Bathrooms',min_value=0, max_value=8,value=2, step = 1)

selected_lot = st.sidebar.number_input('Lot Square Footage',min_value = 500, 
                                       max_value = 1075000, step = 500, value = 14500)

# Load model
model = load_model_ml(fpath = FPATHS['models'][s_model])

if st.checkbox("Generate"):
    
    X_to_pred = get_X_to_predict()
    new_pred = fn.get_prediction(linreg, X_to_pred)
    
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
    
else:
    st.empty()
