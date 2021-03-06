import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.beta_set_page_config(page_title="Heart disease predicter", page_icon="", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Heart disease predicter </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" âšī¸ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.

            """)
        '''
        ## How does it work â 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm đ¨âđž")
        Age = st.number_input("Age (0 - 99)", 0,99)
        Sex = st.number_input("Sex (0 = Male, 1 = Female", 0, 1)
        CPT = st.number_input("Chest Pain Type (0 = No pain, 3 = Tremendeous pain)", 0, 3)
        RBP = st.number_input("Resting Blood Pressure (94 = 200)", 94, 200)
        Chol = st.number_input("Cholesterol (126 = 564)", 126, 564)
        FBS = st.number_input("Fasting Blood Sugar (0 = false, 1 = true)", 0, 1)
        Restecg = st.number_input("Resting Electrocardiographic Results (0 = 2", 0, 2)
        Thalach = st.number_input("Maximum Heart Rate Achieved (71 = 202)", 71, 202)
        Exang = st.number_input("Exercise Induced Angina (0 = no, 1 = yes)", 0, 1)
        Oldpeak = st.number_input("ST depression induced by exercise relative to rest (0.0 - 6.2)", 0.0, 6.2)
        Slope = st.number_input("The slope of the peak exercise ST segment (0 - 2)", 0, 2)
        Ca = st.number_input("Number of major vessels (0 - 3)", 0, 3)
        Thal = st.number_input("3 = normal; 6 = fixed defect; 7 = reversable defect (0 - 7)", 0, 7)


        feature_list = [Age, Sex, CPT, RBP, Chol, FBS, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results đ 
		    ''')
            col1.success(f"{prediction.item().title()}")
      #code for html âī¸ đž đŗ đ¨âđž  đ

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
