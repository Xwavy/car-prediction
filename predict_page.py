import streamlit as st
import pickle
import pandas as pd
import numpy as np
from joblib import load
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

model = load('numodel.joblib')


def show_predict_page():
    st.title("Car Prediction")
    st.write("""The model evaluates
   cars according to the following concept structure:
   CAR=car acceptability
   . PRICE=overall price
   . . buying=buying price
   . . maint=price of the maintenance
   . TECH=technical characteristics
   . . COMFORT=comfort
   . . . doors=number of doors
   . . . persons=capacity in terms of persons to carry
   . . . lug_boot=the size of luggage boot
   . . safety=estimated safety of the car""")
    st.subheader('The predictions are 97% accurate')
    st.write(""" ### We need the following info about the car. 
    When making your selections, use the following 'vhigh'=Very High, 'high'= High, 'med'= Medium, 'low'= Low 
    ### """)

    buy_price = ['vhigh', 'high', 'med', 'low']
    maint_price = ['vhigh', 'high', 'med', 'low']
    doors = st.slider("How many doors does the car have?", 0, 5, 2)
    persons = st.slider("How many people can seat inside the car?", 0, 5, 2)
    lug = ['small', 'med', 'big']
    safety = ['low', 'med', 'high']
    
    df = pd.read_csv('car.data')
    df.replace('?', -99999, inplace=True)
    df = df.dropna()

    buying = st.selectbox("What is the rating of the buying price?", buy_price)
    maint = st.selectbox("What is the maintenance price rating?", maint_price)
    lug_boot = st.selectbox("What is the size of the boot?", lug)
    safety = st.selectbox("What is the safety rating for this car?", safety)

    oe_buying = OrdinalEncoder()
    oe_maint = OrdinalEncoder()
    oe_lugboot = OrdinalEncoder()
    oe_safety = OrdinalEncoder()
    oe_doors = OrdinalEncoder()
    oe_persons = OrdinalEncoder()

    df['buying'] = oe_buying.fit_transform(df[['buying']])
    df['maint'] = oe_maint.fit_transform(df[['maint']])
    df['lug_boot'] = oe_lugboot.fit_transform(df[['lug_boot']])
    df['safety'] = oe_safety.fit_transform(df[['safety']])
    df['doors'] = oe_doors.fit_transform(df[['doors']])
    df['persons'] = oe_persons.fit_transform(df[['persons']])


    ko = st.button('So what kind of car is it?')
    if ko:
        x = np.array([[buying, maint, doors, persons, lug_boot, safety]])
        x[:, 0] = oe_buying.transform(x[:, 0].reshape(1, -1))
        x[:, 1] = oe_maint.transform(x[:, 1].reshape(-1, 1))
        x[:, 2] = oe_doors.transform(x[:, 2].reshape(-1, 1))
        x[:, 3] = oe_persons.transform(x[:, 3].reshape(-1, 1))
        x[:, 4] = oe_lugboot.transform(x[:, 4].reshape(-1, 1))
        x[:, 5] = oe_safety.transform(x[:, 5].reshape(-1, 1))
        res = model.predict(x)
        if res == ['unacc']:
            st.write('This car is unacceptable, and seems to be a bad car.')
        elif res == ['acc']:
            st.write('This car is acceptable.')
        elif res == ['good']:
            st.write('This car is good.')
        else:
            st.write('This car is very good')
