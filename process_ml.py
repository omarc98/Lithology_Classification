def process(las_file, well_data):
    # Plotly imports
    import streamlit as st
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import plotly.express as px
    
    import missingno as msno
    import numpy as np
    from sklearn.impute import SimpleImputer, KNNImputer
    import pandas as pd
    import matplotlib.pyplot as plt
    import facies_plot
    
    st.title('Data Processing')
    
    if not las_file:
        st.warning('No file has been uploaded')  
    else:
        st.write("""The following plot can be used to identify the depth range of each logging curves.
         To zoom in, click and drag on one of the tracks with the left mouse button. 
         To zoom back out double click on the plot.""")

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
            
            
        #LAS Information
        st.subheader('LAS Information')
        rows = well_data.shape[0]
        cols = len(well_data.columns)

        st.write("There are: ",cols,"features and ",rows,"rows")


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

        
        st.subheader(f'Treatment data')
        # User selection for imputation method
        selected_method = st.selectbox("Select method to process the data:", options=["None", "Basic dropna", "Simple Imputer (mean)", "KNN Imputer"])
        rows_before = len(well_data)
        # Display descriptive statistics for the selected method
        if selected_method == "None":
            st.write("No changes were made to the dataframe.")
            imputed_data = well_data     

        elif selected_method == "Basic dropna":
            imputed_data = well_data.copy()
            imputed_data = imputed_data.dropna()
            imputed_data.reset_index(drop=True,inplace=True)
            rows_after = len(imputed_data)
            st.write("Basic dropna")
            st.dataframe(imputed_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write(f'Rows before: {rows_before}\n\n Rows after: {rows_after}')

        elif selected_method == "Simple Imputer (mean)":
            imputer1 = SimpleImputer(strategy='mean')
            imputed_data = well_data.copy()
            imputed_data = pd.DataFrame(imputer1.fit_transform(limpupted_data), columns=imputed_data.columns)
            rows_after = len(imputed_data)
            st.write(f'Rows before: {rows_before}\n\nRows after: {rows_after}')

        else:
            imputed_data = well_data.copy()
            imputer2 = KNNImputer(n_neighbors=5)
            imputed_data = pd.DataFrame(imputer2.fit_transform(imputed_data), columns=imputed_data.columns)
            rows_after = len(imputed_data)
            st.write(f'Rows before: {rows_before}\nRows after: {rows_after}\n')


        st.subheader(f"{selected_method} missingness matrix:")
        fig, ax = plt.subplots(figsize=(12,10))  # Create a figure and axes
        msno.matrix(imputed_data,ax=ax)
        st.pyplot(fig)

 
    #Normalize resistivity curves
    #st.title("Normalization Resistivity Curves")
    #curves_res = st.select('Select the curve/s to normalize', columns)

    #if len(curves_res) < 1:
    #        st.warning('Please select at least 1 curve.')
    #else:
    #    for cur in curves:
    #        imputed_data[cur] = np.log(imputed_data[cur])
    #    st.success("Normalization succesfuly")
    #    st.dataframe(imputed_data[curves_res])
    imputed_data['RLA5'] = np.log(imputed_data['RLA5'])
    #Rename the columns
    dicNamesCurves = {
    "Depth": {
        "nombre" : ["DEPT", "DEPTH", "MD"],
        "default" : "DEPTH"
    },
    "BitSize": {
        "nombre":["BIT", "BITS", "BS", "BitSize"],
        "default": "BS"
    },
    "Caliper": {
        "nombre" : ["CALIPER", "CALIP", "CALI","CAL","DCAL","ACAL","CALA","CALD","CALE","CALH","CALL","CALM","CALML","CALN","CALP","CALS", 
                    "CALT","CALX","CALXH","CALXZ","CALY","CALYH","CALYHD","CALYM","CALYQH","CALYZ","CALZ","CANC","CANN","CAPD","CAX","CAY",
                    "CLDC","CLDM","CLL0","CLMR","CLMS","CLRM","CLS2","CLTC","CLXC","CLXF","CLYC","MCAL","CALXQH","CLCM","CR1","CR2","CS1M",
                    "CS2M","CS3M","CS4M","CS5M","CS6M0","HCA1","HCAL","HCALI","HCALX","HCALY","XCAL","YCAL","CABX","CABY","CACN","CADF","CADP",
                    "CAMR","CAXR","CAYR","DCCP","MLTC","C1","C13","C13A","C13H","C24A","C24","C24H","C24I","C24L","C24M","C24P","C24Z","CA",
                    "CA1","CA2","CADE","CAL1","CAL2","CAL3","CALXM","CALXQ8","CALXGH","CALYQ8","CLCD","CLLO","CQLI","HD1","HD2","HD3","HDAR",
                    "HDIA","HDMI","HDMN","HDMX","HLCA","LCAL","SA","TAC2","C3","HHCA","MBTC","TACC","DZAL","CALIPER", "CALI","CAL","DAC","MSC",
                    "CL","TAC","MCT","EMS","CCT","XYT","CCN","DNSC","DSNCM"],
        "default" : "CALI"
    },
    "Spontaneous Potential": {
        "nombre" : ["SP","IDSP","SPR","SPL","SPDL","SPDHP","SPDH","SPDF","SPD","SPCG","SPC","SP0","SP1","SPBR","SPBRDH","IASP","CGSP","DLSP", 
                    "SPLL"],
        "default" :"SP"
    },
    "Gamma Ray": {
        "nombre" : ["GR","MCG","MGS","NGS","NGT","IPL","GRT","DGR","DG","SL","HDS1","RGD","CWRD","SGR"],
        "default" : "GR"
    },
    "Bulk Density": {
        "nombre" : ["RHOB","RHOZ","DEN","DENB","DENC","DENCDL","HRHO","HRHOB","ZDEN","ZDENS","ZDNCS","ZDNC","HDEN","DENF","DENN",
                    "APLS", "ZDL", "CDL", "SPeD", "SDL","PDS", "MPD","IPL","CDT","LDT","ORD","MDL","DNSC","ASLD"],
        "default" : "RHOB"
    },
    "Neutron Porosity": {
        "nombre" : ["NPHI", "NPH", "CN","DSN","DSEN","MDN","IPL","CNT","CCN","MNP","DNSC","CTN","CN","PHIN","CNC","CNS","HHNPO","HNPHI",
                    "CNCC","CNCD","CNCF","NPRL","TNPH","TPHC","XPOR","NEUT","NeutPor"],
        "default" : "NPHI"
    },
    "Resistivity Shallow": {
        "nombre" : ["LL3","SGRD","SFL","SLL","LLS","RLLS","RLA1","SHALLOW"],
        "default" : "RS"
    },
    "Resistivity Medium": {
        "nombre" : ["R60O","ILM","RILM","RLA3"],
        "default" : "RM"
    },
    "Resistivity Deep": {
        "nombre" : ["R85O","ILD","RILD","DLL","LLD","RLLD","RLA5","DEEP"],
        "default" : "RD"
    },
    "Sonic": {
        "nombre" : ["DT","APX","XMAC","DAL","AC","BCS","DAR","FWS","XACT","CSS","LCS","MSS","UGD","DSI","CST","LST","DNSC","SONIC","BAT"],
        "default" : "DT"
    },
    "Photoelectric Factor":{
        "nombre":["PE","PEF","PEFZ","PDPE","PEDF","PEDN","HPEDN","HPEH8","PE2","PE2QH","PEF8","PEFA","PEFI","PEFL","LPE","PEFS"],
        "default":"PEF"
    }
}
    def rename_columns(df, dicNamesCurves):
        # Crear un diccionario para mapear los nombres actuales a los nombres predeterminados
        rename_map = {}
        # Lista para almacenar los mensajes de cambio
        changes = []

        # Iterar sobre los elementos del diccionario
        for key, value in dicNamesCurves.items():
            default_name = value['default']
            possible_names = value['nombre']

            # Buscar en las columnas del DataFrame si alguna coincide con los nombres posibles
            for col in df.columns:
                if col in possible_names:
                    rename_map[col] = default_name
                    changes.append(f"Curva '{col}' cambiada a '{default_name}' correspondiente a: '{key}'")

        # Renombrar las columnas del DataFrame utilizando el diccionario de mapeo
        df.rename(columns=rename_map, inplace=True)

        # Imprimir los cambios realizados
        for change in changes:
            print(change)

        return df
    
    
    imputed_data = rename_columns(imputed_data,dicNamesCurves)


    # Título de la aplicación
    st.title('LAS File Selection Model and Prediction')

    # Descripción de la aplicación
    st.write("""The following models are used to predict lithology from specific curves of
    LAS file. These are Gamma Ray (GR), Resistivity Deep (RD), Photoelectrical Factor (PEF), 
    Neutron Porosity (NPHI) and Density (RHOB). Please select these specific curves to predict.""")

    # Selección del modelo de clasificación
    selection = st.radio('Select the classification model', ('Machine Learning', 'Neuronal Network'))

    # Verificar que las columnas necesarias están en los datos imputados
    required_columns = ['GR', 'RD', 'PEF', 'NPHI', 'RHOB']
    if all(col in imputed_data.columns for col in required_columns):
        X = imputed_data[required_columns]
    else:
        X = pd.DataFrame(columns=required_columns)  

    # Contenedor inferior para el botón de predicción
    bottom_container = st.container()

    curves_res = required_columns  # Esto es solo un marcador de posición
    
    # Cargar los modelos

    
    # Añadir botón de predicción basado en la selección
    with bottom_container:
        if selection == 'Machine Learning':
            if len(curves_res) == 5:  # Permitir predicción solo con exactamente 5 curvas
                button = st.button("Predict with Machine Learning")
                if button:
                    from joblib import load
                    loaded_ml = load('classification_model.joblib')
                    y_pred = loaded_ml.predict(X)
                    st.write("Predictions:")
                    st.dataframe(y_pred)
                    curves = ['GR','NPHI','RHOB','RD']
                    well_name = 'Pozo con ML'
                    facies_colors = ['#00008B', '#2CA25F', 'gold']
                    imputed_data['Facies_id']=y_pred
                    facies_plot.facies_plot(imputed_data,curves,well_name,facies_colors)
                    
            else:
                st.write("Please select exactly 5 curves to predict with Machine Learning.")
        elif selection == "Neuronal Network":
            if len(curves_res) == 5:  # Permitir predicción solo con exactamente 5 curvas           
                button = st.button("Predict with Neuronal Network")
                if button:
                    from joblib import load
                    from sklearn.preprocessing import StandardScaler
                    scaler= StandardScaler()
                    #loaded_rnn = tf.keras.models.load_model('neuronal_network_model.h5')
                    loaded_rnn = load('neuronal_network_model.joblib')    
                    #X = scaler.fit_transform(X)
                    y_pred = loaded_rnn.predict(X)
                    y_pred = np.argmax(y_pred, axis=1)
                    st.write("Predictions:")
                    st.dataframe(y_pred)
                    curves = ['GR','NPHI','RHOB','RD']
                    well_name = 'Pozo con redes neuronales'
                    facies_colors = ['#00008B', '#2CA25F', 'gold']
                    imputed_data['Facies_id']=y_pred
                    facies_plot.facies_plot(imputed_data,curves,well_name,facies_colors)

