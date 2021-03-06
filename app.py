import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.beta_set_page_config(page_title="Heart Disease Predicter", page_icon="", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Heart Disease Predicter </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
            This site has a vocation to help the doctors in the diagnostic of the patients that probably have heart disease.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict if you probably have a heart disease
        '''


    with col2:
        st.subheader(" Find out if you have a heart disease")
        Age = st.number_input("Age (0 - 99)", 0,99, 0)
        Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1, 0)
        CPT = st.number_input("Chest Pain Type (0 = No pain, 3 = Tremendeous pain)", 0, 3, 0)
        RBP = st.number_input("Resting Blood Pressure (94 = 200)", 94, 200, 94)
        Chol = st.number_input("Cholesterol (126 = 564)", 126, 564, 126)
        FBS = st.number_input("Fasting Blood Sugar (0 = false, 1 = true)", 0, 1, 0)
        Restecg = st.number_input("Resting Electrocardiographic Results (0 = 2", 0, 2, 0)
        Thalach = st.number_input("Maximum Heart Rate Achieved (71 = 202)", 71, 202, 71)
        Exang = st.number_input("Exercise Induced Angina (0 = no, 1 = yes)", 0, 1, 0)
        Oldpeak = st.number_input("ST depression induced by exercise relative to rest (0.0 - 6.2)", 0.0, 6.2, 0.0)
        Slope = st.number_input("The slope of the peak exercise ST segment (0 - 2)", 0, 2, 0)
        Ca = st.number_input("Number of major vessels (0 - 3)", 0, 3, 0)
        Thal = st.number_input("3 = normal; 6 = fixed defect; 7 = reversable defect (0 - 7)", 0, 7, 0)

        feature_list = [Age, Sex, CPT, RBP, Chol, FBS, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results (0 = You probably have no disease - 1 = You probably have a heart disease) 🔍 
		    ''')
            col1.success(f"{prediction.item()}")

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()