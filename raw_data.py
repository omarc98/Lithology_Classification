def raw_data(las_file, well_data):
    import streamlit as st
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import plotly.express as px
    
    import missingno as msno
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    

    st.title('LAS File Data Info')

    if not las_file:
        st.warning('No file has been uploaded')
        return  # Exit the function if no file is uploaded
    #LAS Header
    well_data_header = {
    "Description": [item.descr.capitalize() for item in las_file.well],
    "Mnemonic": [item.mnemonic for item in las_file.well],
    "Value": [item.value for item in las_file.well]}
    df_well = pd.DataFrame(well_data_header)
    st.subheader('**Well Header Information**')
    st.dataframe(df_well)
    
    # Curve Information
    st.subheader('**Curve Information**')
    curves_data = {
    "Curve": [curve.mnemonic for curve in las_file.curves],
    "Units": [curve.unit for curve in las_file.curves],
    "Description": [curve.descr for curve in las_file.curves]}

    # Crear un DataFrame
    df_curves = pd.DataFrame(curves_data)
    st.dataframe(df_curves)

        
    #LAS Information
    st.subheader('LAS Data Visualization')
    rows = well_data.shape[0]
    cols = len(well_data.columns)

    st.write("The LAS file has ",cols,"log curves and ",rows,"data points")

        
    data_nan = well_data.notnull().astype('int')
    # Need to setup an empty list for len check to work
    curves = []
    columns = list(well_data.columns)
    columns.pop(-1) #pop off depth

    col1_md, col2_md= st.columns(2)

    selection = col1_md.radio('Select all data or custom selection', ('All Data', 'Custom Selection'))
    fill_color_md = col2_md.color_picker('Select Fill Colour', '#9D0000')

    if selection == 'All Data':
        curves = columns
    else:
        curves = st.multiselect('Select Curves To Plot', columns)

    if len(curves) <= 1:
        st.warning('Please select at least 2 curves.')
    else:
        curve_index = 1
        fig = make_subplots(rows=1, cols= len(curves), subplot_titles=curves, shared_yaxes=True, horizontal_spacing=0.02)

        for curve in curves:
            fig.add_trace(go.Scatter(x=data_nan[curve], y=well_data['DEPT'], 
            fill='tozerox',line=dict(width=0), fillcolor=fill_color_md), row=1, col=curve_index)
            fig.update_xaxes(range=[0, 1], visible=False)
            fig.update_xaxes(range=[0, 1], visible=False)
            curve_index+=1

        fig.update_layout(height=500, showlegend=False, yaxis={'title':'DEPT','autorange':'reversed'})
        # rotate all the subtitles of 90 degrees
        for annotation in fig['layout']['annotations']: 
                annotation['textangle']=-90
        fig.layout.template='seaborn'
        st.plotly_chart(fig, use_container_width=True)

    # Curve Statistics
    st.subheader('Curve Statistics')
    st.write(well_data.describe())
    
    # Missingness Summary
    st.subheader('Missingness Summary')

    # Calculate percentage of missing values per column
    percent_nan = round(100 * well_data.isnull().sum() / len(well_data), 2)
    percent_nan = percent_nan[percent_nan > 0]  # Filter out columns with 0% missing values
    missing_df = pd.DataFrame({'Curve Name': percent_nan.index, '% Missing': percent_nan.values})
    missing_df = missing_df.style.applymap(lambda x: 'color: red' if x > 10 else '', subset=['% Missing'])


    # Calculate average missingness across columns with missing values
    mean_percent = round(percent_nan.mean(), 2) if not percent_nan.empty else 0

    # Identify columns with high missingness (> 10%)
    columns_with_nulls = list(percent_nan.loc[percent_nan > 10].index)

    if not percent_nan.empty:
        # Display percentage of missing values
        st.subheader('**Percentage of null data in each column:**')
        st.write(missing_df)

        # Display columns with high missingness
        if columns_with_nulls:
            st.warning(f'Columns with data missing in more than 10% of rows:')
            st.write(', '.join(columns_with_nulls))
            st.info("Suggestion: Please review the curve information to apply a filling method ")
        else:
            st.write('No columns have more than 10% missing data.')
    else:
        st.write('There are no missing values in the data.')




