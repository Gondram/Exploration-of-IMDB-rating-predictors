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

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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

@st.cache_data
def load_network(fpath):
    model = tf.keras.models.load_model(fpath)
    return model

@st.cache_data
def load_lookup(fpath=FPATHS['data']['ml']['target_lookup']):
    return joblib.load(fpath)

def predict_decode_deep(X_to_pred, network,lookup_dict,
                       return_index=True):
    
    if isinstance(X_to_pred, str):
        
        X = [X_to_pred]
    else:
        X = X_to_pred
    
    pred_probs = network.predict(X)

    pred_class = fn.convert_y_to_sklearn_classes(pred_probs)
    
    # Decode label
    class_name = lookup_dict[pred_class[0]]

    return class_name


def classification_metrics_streamlit(y_true, y_pred, label='',
                           figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f",
                                    class_names=None):
    """Modified version of classification metrics function from Intro to Machine Learning.
    Updates:
    - Reversed raw counts confusion matrix cmap  (so darker==more).
    - Added arg for normalized confusion matrix values_format
    """
    # Get the classification report
    report = classification_report(y_true, y_pred,target_names=class_names)
    
    ## Save header and report
    header = "-"*70
    final_report = "\n".join([header,f" Classification Metrics: {label}", header,report,"\n"])
        
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            display_labels = class_names, # Added display labels
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            display_labels = class_names, # Added display labels
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()

    return final_report, fig

def classification_metrics_streamlit_tensorflow(model,X_train=None, y_train=None, 
                                                label='Training Data',
                                    figsize=(6,4), normalize='true',
                                    output_dict = False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                                class_names = None,
                                    colorbar=False):
    
    ## Check if X_train is a dataset
    if hasattr(X_train,'map'):
        # If it IS a Datset:
        # extract y_train and y_train_pred with helper function
        y_train, y_train_pred = fn.get_true_pred_labels(model, X_train)
    else:
        # Get predictions for training data
        y_train_pred = model.predict(X_train)


     ## Pass both y-vars through helper compatibility function
    y_train = fn.convert_y_to_sklearn_classes(y_train)
    y_train_pred = fn.convert_y_to_sklearn_classes(y_train_pred)
    
    # Call the helper function to obtain regression metrics for training data
    report, conf_mat = classification_metrics_streamlit(y_train, y_train_pred, 
                                                        figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_train, 
                                                        values_format=values_format,label=label,
                                                       class_names=class_names)
    return report, conf_mat

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

X_to_pred = st.sidebar.text_input('Insert Text', value = 'Great Film!')

# Load model
model = load_model_ml(fpath = FPATHS['models'][s_model])

if st.button("Generate"):
    if s_model == 'nlp':
        
        fpath_model = FPATHS['models']['nlp'] # Permissions issue, I just gotta get this working
        # fpath_model = 'models\nlp-pipe.keras'
        best_network = load_network(fpath_model)
        target_lookup = load_lookup()
        pred_class_name = predict_decode_deep(X_to_pred, best_network,target_lookup)
        st.markdown(f"##### Neural Network Predicted category:  {pred_class_name}")
        
    else:
        fpath_model = FPATHS['models']['ml']

    new_pred = fn.get_prediction(model, X_to_pred)
    
    st.markdown(f"> #### {s_model} Model Predicted Rating = {new_pred}")
    
else:
    st.empty()

st.divider()

st.subheader('Evaluate Neural Network')

## To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)

if s_model == 'nlp':
    if st.button("Show Evaluation."):
        with st.spinner("Please wait while the neural network is evaluated..."):
            if show_train == True:
                # Display training data results
                report_str, conf_mat = classification_metrics_streamlit_tensorflow(best_network,label='Training Data',
                                                                                   X_train=train_ds,
                                                                                   )
                st.text(report_str)
                st.pyplot(conf_mat)
                st.text("\n\n")
    
            if show_test == True: 
                # Display training data results
                report_str, conf_mat = classification_metrics_streamlit_tensorflow(best_network,label='Test Data',
                                                                                   X_train=test_ds
                                                                               )
                st.text(report_str)
                st.pyplot(conf_mat)
                st.text("\n\n")
  
    else:
        st.empty()

else: 
    if st.button("Show Evaluation."):
        with st.spinner("Please wait while the neural network is evaluated..."):
            st.empty()




