import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image

@st.cache_data
def load_data():
    df = pd.read_csv('car.data')
    df = df.dropna()
    df.replace('?', -99999, inplace=True)
    return df


def show_explore_page():
    st.title('Car Performance Prediction Model By Philip Olufunmilayo')
    st.header('Information on the Dataset')
    st.write("""
    Relevant Information Paragraph:

    Car Evaluation Database was derived from a simple hierarchical
    decision model originally developed for the demonstration of DEX
    (M. Bohanec, V. Rajkovic: Expert system for decision
    making. Sistemica 1(1), pp. 145-157, 1990.). The model evaluates
    cars according to the following concept structure:

    CAR                      car acceptability
    . PRICE                  overall price
    . . buying               buying price
    . . maint                price of the maintenance
    . TECH                   technical characteristics
    . . COMFORT              comfort
    . . . doors              number of doors
    . . . persons            capacity in terms of persons to carry
    . . . lug_boot           the size of luggage boot
    . . safety               estimated safety of the car

    Input attributes are printed in lowercase. Besides the target
    concept (CAR), the model includes three intermediate concepts:
    PRICE, TECH, COMFORT. Every concept is in the original model
    related to its lower level descendants by a set of examples (for
    these examples sets see http://www-ai.ijs.si/BlazZupan/car.html).
    
    The Car Evaluation Database contains examples with the structural
    information removed, i.e., directly relates CAR to the six input
    attributes: buying, maint, doors, persons, lug_boot, safety.
    
    Because of known underlying concept structure, this database may be
    particularly useful for testing constructive induction and
    structure discovery methods.
    
    5. Number of Instances: 1728
    (instances completely cover the attribute space)
    
    6. Number of Attributes: 6
    
    7. Attribute Values:
    
    buying       v-high, high, med, low
    maint        v-high, high, med, low
    doors        2, 3, 4, 5-more
    persons      2, 4, more
    lug_boot     small, med, big
    safety       low, med, high
    
    8. Missing Attribute Values: none
    
    9. Class Distribution (number of instances per class)
    
    unacc     1210     (70.023 %) 
    acc        384     (22.222 %) 
    good        69     ( 3.993 %) 
    v-good      65     ( 3.762 %)
    
    """)
    st.subheader('This shows the correlation of the features in the dataset')
    img = Image.open('correlation.png')
    st.image(img, caption='correlation of the features')
    st.subheader('This shows the disribution of the classes of cars in the dataset')
    img1 = Image.open('distribution.png')
    st.image(img1, caption='disribution of the classes of cars in the dataset')


