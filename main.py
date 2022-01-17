
# Import libraries
import pandas as pd
import numpy as np
import streamlit as st

import seaborn as sb
sb.set_style("ticks")

import kmodes
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import random
import time

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt



# Radio
rad = st.sidebar.radio('Pages:',['About KPrototypes','Data Clustering'])

# About KPrototypes ************************************************************************
if rad == 'About KPrototypes':
    st.header('Clustering Mixed Data Types Using K-Prototypes')
    st.image('cluster.jpg', width=700)
    st.markdown(
        """This web app finds the clusters of a dataset made up of both numeric and categorical data using
        **K-Prototypes**. It is a combination of both **K-means** (for numeric features) and **K-modes** (for
         categorical features). **K-means** utilizes the euclidean distance to compute the clusters--the lesser
          the distance between points, the more similar the data points are. It is derived as follows:""")

    st.image('euclidean.png')

    st.markdown(
        """For **K-modes**, it clusters categorical (non-numeric) data by using dissimilarities between data points.
        Similar to **K-means**, the lesser the dissimilarities, the more similar the data points are. As the name
        suggests, **K-modes** uses the mode instead of the mean.
        """)

    st.markdown(
        """On the sidebar at the left, click **Data Clustering** to proceed with the analysis.
        """)



# Data Clustering ************************************************************************

if rad == 'Data Clustering':

    # Step 1: Step 1: Upload Data and Visualize
    st.header('Step 1: Upload Data and Visualize')

    st.markdown("""
    This web app allows you to upload your own dataset in csv format which may consists of several numeric and/or 
    categorical features. Note that a dummy dataset composed of numeric and categorical data is used 
    below prior to your dataset upload.""")

    st.write('')

    st.markdown("""
        Click on the **Browse files** button below to upload your data.""")

    my_file = st.file_uploader('Upload Dataset')


    if my_file:
        df = pd.read_csv(my_file)
    else:

        # dummy dataset
        x = random.sample(range(1,500), 100)
        y = random.sample(range(1,500), 100)
        a = random.sample(range(1,500), 100)
        my_z = ['NCR','Luzon','Visayas','Mindanao']
        my_r = ['Single','Married','Widow','Uknown']
        z = random.choices(my_z, k=100)
        r = random.choices(my_r, k=100)
        df = pd.DataFrame()
        df['x'] = x
        df['y'] = y
        df['a'] = a
        df['z'] = z
        df['r'] = r



    st.markdown("""
        The uploaded dataset is shown below.""")
    st.dataframe(df)

    # select numeric and categorical cols
    my_cat = df.select_dtypes(exclude='number').columns
    my_num = list(set(df.columns) - set(my_cat))



    # standardize
    st.write('')
    st.write('')
    st.write('')
    st.markdown("""
            The **StandardScaler** is used to standardize numerical features. The standardized dataset is shown below.""")

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[my_num]))
    df_scaled.columns = my_num
    df = pd.merge(df_scaled, df[my_cat],  how='left', left_index=True, right_index=True)
    st.dataframe(df)


    # Step 2: Scatterplot of Scaled Numeric Data
    st.header('Step 2: Scatterplot of Scaled Numeric Data')

    st.markdown("""
        Select the numeric columns that you would like to visualize.""")

    my_vars1 = st.selectbox('Select first numeric columns to visualize:', my_num)
    my_vars2 = st.selectbox('Select second numeric columns to visualize:', my_num)
    my_vars3 = st.selectbox('Select hue dimension using categorical column:', my_cat)

    sb.scatterplot(data=df, x=my_vars1, y=my_vars2, hue=my_vars3)
    plt.title('Scatterplot of Selected Numeric Columns (Scaled)')
    plt.legend(loc='upper left')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Step 3: K-Prototypes
    st.header('Step 3: K-Prototypes')


    # select number of clusters
    st.markdown("""
                **K-Prototypes** will now be initiated for the data. To define the parameter of the model, 
                select the number of clusters to be used by using the slider. The number of clusters would define
                the number of groups the data points""")



    n_clusters = st.slider('Select the number of clusters:', 2, 15)
    my_cat_cols = [df.columns.get_loc(i) for i in my_cat if i in df]

    kproto = KPrototypes(n_clusters=n_clusters, init='Cao')
    clusters = kproto.fit_predict(df, categorical= my_cat_cols)
    df['clusters'] = clusters
    df['clusters'] = df['clusters'].apply(lambda x: 'cluster' + str(x))

    st.dataframe(df)

    st.write('')
    st.write('')
    st.markdown("""
            The dataframe has been updated with the **clusters** column above. Below, visualize the data further by 
            selecting the numeric columns and the **clusters** as the color (hue). Notice the groupings of
            the data points (i.e. data with similar colors are clustered together) based on the numeric dimensions that you chose. To further visualize the impact
            of the number of clusters parameter, increase or decrease the slider above to define the number
            of clusters to be used.""")

    # select numeric and categorical cols
    my_cat2 = df.select_dtypes(exclude='number').columns
    my_num2 = list(set(df.columns) - set(my_cat2))

    my_vars4 = st.selectbox('Select first numeric columns to visualize:', my_num2, key='cluster')
    my_vars5 = st.selectbox('Select second numeric columns to visualize:', my_num2, key='cluster')
    my_vars6 = st.selectbox('Select hue dimension using categorical column:', my_cat2, key='cluster')

    sb.scatterplot(data=df, x=my_vars4, y=my_vars5, hue=my_vars6)
    plt.title('Scatterplot of Clustered Data')
    plt.legend(loc='upper left')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



    # Step 4: Elbow Method

    st.header('Step 4: The Elbow Method')
    st.write('')
    st.write('')
    st.markdown("""
                    Click the **Calculate Optimal Number of Clusters** button below to show the 
                    **Elbow Method** graph and visualize the optimal number of clusters.""")

    clust_button = st.button('Calculate Optimal Number of Clusters')

    if clust_button:
        progress = st.progress(0)
        my_status = 'Running calculations...'
        st.write(my_status)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)


        cost = []
        for num_clusters in list(range(1, 8)):
            kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
            kproto.fit_predict(df.drop(['clusters'], axis=1), categorical=my_cat_cols)
            cost.append(kproto.cost_)

        plt.plot(cost)
        plt.title('Elbow Method - Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Epsilon (sum of squares of distance between data points)')
        st.pyplot()
        st.write('')
        st.write('')
        st.markdown("""
                            Note the inflection point in the graph which shows the optimal number of clusters to be used.""")

    else:
        pass








