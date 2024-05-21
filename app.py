import streamlit as st
from io import StringIO

import raw_data
import plot
import process_ml
import facies_plot

@st.cache_data
def load_data(uploaded_file):
    import lasio
    import pandas as pd
    from io import StringIO
    import streamlit as st
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
    st.sidebar.markdown(f'<b>Well Name: </b>: {las_file.well.WELL.value}',unsafe_allow_html=True)
    
def home():
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">Lithology Classification App</h1>
        <h2>Welcome to the Lithology Classification</h2>
        <h4 style="text-align: center;">Created by Omar Cruz and Luis Cosquillo</h4>
        <p style="font-size: 18px; line-height: 1.6;">
            Lithology classification is an app that applies machine learning models and neural networks
            to reduce interpretation time, providing a prognosis of formations in the field.
        </p>
        <p style="font-size: 18px; line-height: 1.6;">
            To begin using the app, load your LAS file using the file upload option on the sidebar. 
            Once you have done this, you can navigate to the relevant tools using the Navigation menu.
        </p>
        <h2>Sections</h2>
        <ul style="font-size: 18px; line-height: 1.6; text-align: left;">
            <li><strong>Data Information:</strong> Information about the curves contained within the LAS file, including names, statistics, and raw data values.</li>
            <li><strong>Data Visualisation:</strong> Visualisation tools to view LAS file data on a log plot, crossplot, and histogram.</li>
            <li><strong>Data Processing and ML:</strong> Process the data to eliminate null values or possible anomalous data before predicting using machine learning or neural networks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
#Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', 
    ['Home', 'Data Information', 'Data Visualisation', 'Data Processing and ML'])

if options == 'Home':
    home()
elif options == 'Data Information':
    raw_data.raw_data(las_file, well_data)
elif options == 'Data Visualisation':
    plot.plot(las_file, well_data)
elif options == 'Data Processing and ML':
    process_ml.process(las_file,well_data)
else:
    st.sidebar.error('Please upload a LAS file to proceed.')


