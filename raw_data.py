def raw_data(las_file, well_data):
    import streamlit as st
    import missingno as msno
    import matplotlib.pyplot as plt 
    """Displays LAS file data information and missingness matrix in Streamlit."""

    st.title('LAS File Data Info')

    if not las_file:
        st.warning('No file has been uploaded')
        return  # Exit the function if no file is uploaded

    # Curve Information
    st.subheader('**Curve Information**')
    for count, curve in enumerate(las_file.curves):
        st.write(f"Curve: {curve.mnemonic}, \t\t Units: {curve.unit}, \t Description: {curve.descr}")
        
    #LAS Information
    st.subheader('LAS Information')
    rows = well_data.shape[0]
    cols = len(well_data.columns)

    st.write("There are: ",cols,"features and ",rows,"rows")

        
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

        fig.update_layout(height=1000, showlegend=False, yaxis={'title':'DEPT','autorange':'reversed'})
        # rotate all the subtitles of 90 degrees
        for annotation in fig['layout']['annotations']: 
                annotation['textangle']=-90
        fig.layout.template='seaborn'
        st.plotly_chart(fig, use_container_width=True)

    # Curve Statistics
    st.write('<b>Curve Statistics</b>', unsafe_allow_html=True)
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
        st.write('**Percentage of null data in each column:**')
        st.write(missing_df)

        # Display columns with high missingness
        if columns_with_nulls:
            st.subheader(f'Columns with data missing in more than 10% of rows:')
            st.write(', '.join(columns_with_nulls))
        else:
            st.write('No columns have more than 10% missing data.')
    else:
        st.write('There are no missing values in the data.')




