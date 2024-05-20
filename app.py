import lasio
import missingno as mno
import pandas as pd
import streamlit as st
from io import StringIO

import header
import raw_data
import plot
import process_ml
import facies_plot

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            str_io = StringIO(bytes_data.decode('Windows-1252'))
            las_file = lasio.read(str_io)
            well_data = las_file.df()
            well_data['DEPT'] = well_data.index
            return las_file, well_data

        except UnicodeDecodeError as e:
            st.error(f"error loading log.las: {e}")
            return None, None
    
    else:
        return None, None
### File Upload ###
las_file=None
well_data=None

st.sidebar.write('# LAS Data Explorer')
st.sidebar.write('To begin using the app, load your LAS file using the file upload option below.')

uploadedfile = st.sidebar.file_uploader(' ', type=['.las'])
las_file, well_data = load_data(uploadedfile)

if las_file:
    st.sidebar.success('File Uploaded Successfully')
    st.sidebar.write(f'<b>Well Name: </b>: {las_file.well.WELL.value}',unsafe_allow_html=True)
    
def home():
    st.title('Lithology Classification App')
    st.write('## Welcome to the Lithology Classification')
    st.write('#### Created by Omar Cruz and Luis Cosquillo')
    st.write('''Lithology classification is an app that applies machine learning models       and neural networks to reduce interpretation time, providing a prognosis of formations     in the field.''')
    st.write('To begin using the app, load your LAS file using the file upload option on       the sidebar. Once you have done this, you can navigate to the relevant tools using the     Navigation menu.')
    st.write('\n')
    st.write('## Sections')
    st.write('**Header Info:** Information from the LAS file header.')
    st.write('**Data Information:** Information about the curves contained within the LAS       file, including names, statisics and raw data values.')
    st.write('**Data Visualisation:** Visualisation tools to view las file data on a log       plot, crossplot and histogram.')
    st.write('**Data Processing and ML:** Process the data to eliminate null values or possible anomalous data before predicting using machine learning or neural networks.')
    
#Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', 
    ['Home', 'Header Information', 'Data Information', 'Data Visualisation', 'Data Processing and ML'])

if options == 'Home':
    home()
elif options == 'Header Information':
    header.header(las_file)
elif options == 'Data Information':
    raw_data.raw_data(las_file, well_data)
elif options == 'Data Visualisation':
    plot.plot(las_file, well_data)
elif options == 'Data Processing and ML':
    process_ml.process(las_file,well_data)

    
# Mensaje de error si no se ha subido un archivo LAS
if options != 'Home' and not las_file:
    st.sidebar.error('Please upload a LAS file to proceed.')


