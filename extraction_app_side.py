# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:45:41 2020

@author: rejid4996
"""

import streamlit as st
from streamlit.server.Server import Server
import pandas as pd
import os
import numpy as np
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
from sklearn import preprocessing
import re
#import spacy
#from spacy.lang.en import English
#from spacy import displacy
#nlp = spacy.load('en_core_web_sm')
import logging
logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

def main():
    """NLP App with Streamlit"""
    
    from PIL import Image
    logo = Image.open('ArcadisLogo.jpg')
    logo = logo.resize((300,90))
    
    st.sidebar.image(logo)
    st.sidebar.title("AI for Site Summaries")
    st.sidebar.subheader("Text extraction using Elmo ")
    
    st.info("Automation of Site Summaries using Natural Language Processing")
    
    uploaded_file = st.sidebar.file_uploader("Choose the Knowledge base file", type="xlsx")
    


    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        
        search_string = st.sidebar.text_input("", "your search word...")
        
        gcr_config = st.sidebar.slider(label="choose the no of Sentences",
                           min_value=1,
                           max_value=10,
                           step=1)
    
        run_button = st.sidebar.button(label='Run Extraction')
        
        # create sentence embeddings
        url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(url)
        
        text = ' '.join(df.Sentences)
        text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ')
        text = ' '.join(text.split())
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
        
        
        embeddings = embed(
            sentences,
            signature="default",
            as_dict=True)["default"]
        
        
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          x = sess.run(embeddings)
  
        
        pca = PCA(n_components=50)
        y = pca.fit_transform(x)
        

        
        y = TSNE(n_components=2).fit_transform(y)
        
        data = [
            go.Scatter(
                x=[i[0] for i in y],
                y=[i[1] for i in y],
                mode='markers',
                text=[i for i in sentences],
            marker=dict(
                size=16,
                color = [len(i) for i in sentences], #set color equal to a variable
                opacity= 0.8,
                colorscale='viridis',
                showscale=False
            )
            )
        ]
        layout = go.Layout()
        layout = dict(
                      yaxis = dict(zeroline = False),
                      xaxis = dict(zeroline = False)
                     )
        fig = go.Figure(data=data, layout=layout)
        
        
        #search_string = "soil contamination"
        #results_returned = "5"
        results_returned = gcr_config
        
        embeddings2 = embed(
            [search_string],
            signature="default",
            as_dict=True)["default"]
        
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          search_vect = sess.run(embeddings2)
         
        cosine_similarities = pd.Series(cosine_similarity(search_vect, x).flatten())

        output =""
        for i,j in cosine_similarities.nlargest(int(results_returned)).iteritems():
          for i in sentences[i].split():
            if i.lower() in search_string:
              output += " "+str(i)+ ","
            else:
              output += " "+str(i)
        
        output_list = list(output.split(".")) 
        
        output1 = pd.DataFrame(output_list, columns = ['extracted text'])
        output1.dropna()
        
        st.table(output1)
        
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            val = to_excel(df)
            b64 = base64.b64encode(val)  # val looks like b'...'
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download file</a>' # decode b'abc' => abc

        st.markdown(get_table_download_link(output1), unsafe_allow_html=True)
        
        st.success("Rather than a dictionary of words and their corresponding vectors, ELMo analyses words within the context that they are used. It is also character-based, allowing the model to form representations of out-of-vocabulary words.")

        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()
