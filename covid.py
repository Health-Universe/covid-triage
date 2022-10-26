# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:47:17 2022

@author: chags
"""

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pycaret.classification import setup, create_model, predict_model, get_config
import plotly.express as px
import shap
from streamlit_shap import st_shap
from streamlit_echarts import st_echarts

#use pandas to read covid data for model training and creation
train = pd.read_csv('COVID_TRAIN.csv')
#use pycaret to preprocess and train a decision tree supervised ML model
exp = setup(train, target = 'class', silent=True, verbose=False)
dt = create_model('dt')
#set settings for streamlit page
st.set_page_config(layout="wide",
    page_title="COVID Triage",
    page_icon="chart_with_upwards_trend")

#hide streamlit menu bar
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#option menu from streamlit component streamlit_option_menu
with st.sidebar:
    selected = option_menu(None, ["Log In", "COVID Triage"], 
    icons=['house',  "list-task"], 
    menu_icon="cast", default_index=0, orientation="vertical")
#log in page --- no real functionality -- just for looks
if selected == 'Log In':
    st.title('Covid 19 Mortality Risk Clinical Decision Support System')
    if 'user_name' not in st.session_state:
        st.session_state.user_name = 'User Name'
    user_name = st.text_input("User Name", value = st.session_state.user_name)
    st.session_state.user_name = user_name
    password = st.text_input("Password", type="password")
    st.session_state.password = password
    if st.session_state.password:
        sc = "Welcome " + st.session_state.user_name + " please proceed to COVID Triage"
        st.success(sc)
#covid triage page
#feature inputs correspond to training data
if selected == 'COVID Triage':
    col1, col2, col3= st.columns(3)
    with col1:
        name = st.text_input("Patient ID")
    with col2:
        gender = st.selectbox("Gender", (0, 1), help = "0 = Female, 1 = Male")
    with col3:
        age = st.number_input("Age", step = 1)

    ccol1, ccol2, ccol3, ccol4= st.columns(4)
    with ccol1:
        bun = st.slider("Blood Urea Nitrogen", min_value=0, max_value=60)
    with ccol2:
        inr = st.slider("International Normalized Ratio", min_value=0.88, max_value=1.66)
    with ccol3:
        honeuro = st.selectbox("History of Neurological Disorder", (0, 1), help = "0 = No history of neurological disorders, 1 = History of neurological disorders")
    with ccol4:
        hocard = st.selectbox("History of Cardiovascular Disorder", (0, 1), help = "0 = No history of cardiovascular disorders, 1 = History of cardiovascular disorders")
        
    test = pd.DataFrame({"Age": [age], "Gender":[int(gender)],"Blood Urea Nitrogen":[bun], "Cardiovascular History":[int(hocard)], "Neurological History":[int(honeuro)], "Int Norm Ratio":[inr]})

    preds = predict_model(dt, test)
    st.sidebar.text('Risk Prediction and Confidence')
    preds['Mortality Risk'] = preds['Label'].replace([0,1], ['Low Mortality Risk', 'High Mortality Risk'])
    if preds['Label'].iloc[0] == 0:
        #display if label = 0 (low mortality risk)
        st.sidebar.info(preds['Mortality Risk'].iloc[0])
        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [preds['Score'].iloc[0]]}]
        }
    if preds['Label'].iloc[0] == 1:
        #display if label = 1 (high mortality risk)
        st.sidebar.error(preds['Mortality Risk'].iloc[0])
        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [preds['Score'].iloc[0]], 'color': ['#ff0000']}]
        }
    with st.sidebar:
        #liquid fill chart with confidence value and color corresponding to mortality risk (high = red, low = blue)
        st_echarts(liquidfill_option)
    #shapley additive explanation of feature weights on model prediction
    explainer = shap.KernelExplainer(model = dt.predict_proba, data = get_config('X_train'), link = "identity")
    shap_value_single = explainer.shap_values(X = test)
    st_shap(shap.force_plot(base_value = explainer.expected_value[0],
                    shap_values = shap_value_single[0], features=test
                    ))
    #show previous patients and how the current patient compares through histograms
    st.text("Previous COVID-19 ICU Patients")
    df = train.copy()
    for cols in df.drop(['Unnamed: 0', 'class'], axis=1).columns:
        df['Mortality Outcome'] = train['class'].replace([1,0], ['Deceased', 'Survived'])
        fig = px.histogram(df, x = cols, color = 'Mortality Outcome',
                           color_discrete_map = {'Deceased':'red','Survived':'blue'})
        fig.add_vline(x=test[cols].iloc[0], line_dash="dot",
                      annotation_text="Current Patient", 
                      annotation_position="top left",
                      annotation_font_size=10,
                      annotation_font_color="gray"
                     )
        st.plotly_chart(fig, config= dict(
    displayModeBar = False))
        
