import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import sys
import time
import datetime
import json



pipe_lr=joblib.load(open('models/emotion_classifier.pkl',"rb"))


def predict_emotions(docs):
    results=pipe_lr.predict([docs])
    return results[0]


def main():
    st.title("Emotion Prediction")
    menu=['Home','Monitoring','About']
    choice=st.sidebar.selectbox('Menu',menu)

    if choice=='Home':
        st.subheader('Home-Emotion in Text')

        with st.form(key='Emotion_form'):
            st.text('Enter the text below')
            text=st.text_area('Type here')
            submit=st.form_submit_button(label='Submit')
        if submit:
            col1,col2=st.beta_columns(2)
            predictions=predict_emotions(text)
            with col1:
                st.success('Original')
                st.write(text)
                st.success('Predictions')
                st.write(predictions)
            with col2:
                st.success('Precition Probability')
                    

    elif choice=='Monitoring':
        st.subheader('Monitoring App')

    else:
        st.subheader('About')

if __name__ == '__main__':
    main()