import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score,confusion_matrix

st.write("""
# Bu Mantar Yenilebilir Mi ? 🤔🧐

Girdiğiniz değerlere göre mantarın zehirli olup olmadığını tahmin eden bir uygulama!
""")

mushrooms=pd.read_csv("mushrooms.csv")

le=LabelEncoder()
le_bruises = LabelEncoder()
le_gill_spacing = LabelEncoder()
le_gill_size = LabelEncoder()
le_gill_color = LabelEncoder()
le_ring_type = LabelEncoder()

for i in mushrooms.columns:
    mushrooms[i]=le.fit_transform(mushrooms[i])

mushrooms.drop('veil-type',inplace=True,axis=1)

selected_columns = ['bruises', 'gill-spacing', 'gill-size', 'gill-color', 'ring-type']
x = mushrooms[selected_columns]
y=mushrooms.iloc[:,0]


X=x.values
Y=y.values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

classifier_name = st.sidebar.selectbox("Algoritma Seç",("Random Forest","Desicion Tree"))
st.sidebar.write("---")


st.sidebar.header('Parametreleri Girin')

file=open("params.txt","r",encoding="utf-8")
fb=file.readline()
fgsp=file.readline()
fgsz=file.readline()
fgc=file.readline()
frt=file.readline()

def user_input_features(fb, fgsp, fgsz, fgc, frt):
    bruises = st.sidebar.selectbox(fb.split(':')[0], (fb.split(':')[1].split(',')[0], fb.split(':')[1].split(',')[1]))
    gill_spacing = st.sidebar.selectbox(fgsp.split(':')[0], (fgsp.split(':')[1].split(',')[0], fgsp.split(':')[1].split(',')[1], fgsp.split(':')[1].split(',')[2]))
    gill_size = st.sidebar.selectbox(fgsz.split(':')[0], (fgsz.split(':')[1].split(',')[0], fgsz.split(':')[1].split(',')[1]))
    gill_color = st.sidebar.selectbox(fgc.split(':')[0], (fgc.split(':')[1].split(',')[0], fgc.split(':')[1].split(',')[1], fgc.split(':')[1].split(',')[2], fgc.split(':')[1].split(',')[3], fgc.split(':')[1].split(',')[4], fgc.split(':')[1].split(',')[5], fgc.split(':')[1].split(',')[6], fgc.split(':')[1].split(',')[7], fgc.split(':')[1].split(',')[8], fgc.split(':')[1].split(',')[9], fgc.split(':')[1].split(',')[10], fgc.split(':')[1].split(',')[11]))
    ring_type = st.sidebar.selectbox(frt.split(':')[0], (frt.split(':')[1].split(',')[0], frt.split(':')[1].split(',')[1], frt.split(':')[1].split(',')[2], frt.split(':')[1].split(',')[3], frt.split(':')[1].split(',')[4], frt.split(':')[1].split(',')[5], frt.split(':')[1].split(',')[6], frt.split(':')[1].split(',')[7]))

    bruises_input_encoded = le_bruises.fit_transform([bruises.split('=')[1]])
    gill_spacing_input_encoded = le_gill_spacing.fit_transform([gill_spacing.split('=')[1]])
    gill_size_input_encoded = le_gill_size.fit_transform([gill_size.split('=')[1]])
    gill_color_input_encoded = le_gill_color.fit_transform([gill_color.split('=')[1]])
    ring_type_input_encoded = le_ring_type.fit_transform([ring_type.split('=')[1]]),
    data = {
    'bruises': bruises_input_encoded[0],
    'gill_spacing': gill_spacing_input_encoded[0],
    'gill_size': gill_size_input_encoded[0],
    'gill_color': gill_color_input_encoded[0],
    'ring_type': ring_type_input_encoded[0],
    }   
    user_input_df = pd.DataFrame(data, index=[0])


    return user_input_df

df = user_input_features(fb, fgsp, fgsz, fgc, frt)

def get_predict(clf_name, user_input_df):
    if clf_name == "Random Forest":
        rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
    else:
        dtc = DecisionTreeClassifier(criterion="entropy")
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

    user_input = user_input_df.values
    user_prediction = rfc.predict(user_input) if clf_name == "Random Forest" else dtc.predict(user_input)

    return y_pred, cm, user_prediction

y_pred, cm, user_prediction = get_predict(classifier_name, df)

result = "Zehirli ☠️" if user_prediction[0] == 1 else "Normal 🍄"

st.subheader(f"Tahmin : {result}")

total=0

for i in range(2):
    for j in range(2):
        total+=cm[i][j]

probability=((cm[0][0]+cm[1][1])/total)*100

st.subheader("Olasılıklar:")

if user_prediction[0] == 1:
    st.write("Zehirli olma olasılığı : " + str(probability) + "%")
    st.write("Normal olma olasılığı : " + str(100 - probability) + "%")
else:
    st.write("Normal olma olasılığı : " + str(probability) + "%")
    st.write("Zehirli olma olasılığı : " + str(100 - probability) + "%")

# random inputs