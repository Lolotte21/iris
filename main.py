import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris.
''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('Longueur du Sépale', 4.3,7.9, 5.3)
    sepal_width=st.sidebar.slider('Largeur du Sépale', 2.0,4.4, 3.3)
    petal_length=st.sidebar.slider('Longueur du Pétale', 1.0,6.9, 2.3)
    petal_width=st.sidebar.slider('Largeur du Sépale', 0.1,2.5, 1.3)
    data={'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width
    } 
    fleur_parametres=pd.DataFrame(data, index=[0])
    return fleur_parametres

df=user_input()

st.subheader('Rappel de vos paramètres :')
st.write(df)

iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data, iris.target)

prediction=clf.predict(df)

st.subheader('La catégorie de fleur d\'Iris est :')
col1, col2 = st.columns([20,25])
with col1:
    st.write(iris.target_names[prediction])
with col2:
    st.image("img/Z2.png", width=424)
