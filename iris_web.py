# Libraries
from os import waitpid
import pickle

from sklearn import linear_model
import streamlit as st
import pandas as pd

# Load models
with open('linear_regression.pkl', 'rb') as li:
    lr_model = pickle.load(li)

with open('logistic_regression.pkl', 'rb') as lg:
    lg_model = pickle.load(lg)

with open('svc.pkl', 'rb') as svc:
    svc_model = pickle.load(svc)

def classify(number):
    if number == 0:
        return 'Setosa'
    elif number == 1:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    # Heading Title
    st.title('Modelamiento de Iris')

    # Heading Sidebar Title
    st.sidebar.header('User Input Parameters')

    # Sidebar Function Parameters
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
                
        data = {'sepal_length':sepal_length,
                'sepal_width':sepal_width,
                'petal_length':petal_length,
                'petal_width':petal_width}
        
        features = pd.DataFrame(data, index=[0])

        return features
    
    df = user_input_parameters()

    option = ['Linear Regression', 'Logistic Regression', 'SVC']

    model = st.sidebar.selectbox('Which model you like to use', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lr_model.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(lg_model.predict(df)))
        else:
            st.success(classify(svc_model.predict(df)))


if __name__ == '__main__':
    main()