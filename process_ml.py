def process(las_file, well_data):
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
            st.write("Stadistics before treatment data")
            st.dataframe(well_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write("Basic dropna")
            st.dataframe(imputed_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write(f'Rows before: {rows_before}\n\n Rows after: {rows_after}')

        elif selected_method == "Simple Imputer (mean)":
            imputer1 = SimpleImputer(strategy='mean')
            imputed_data = well_data.copy()
            imputed_data = pd.DataFrame(imputer1.fit_transform(limpupted_data), columns=imputed_data.columns)
            rows_after = len(imputed_data)
            st.write("Stadistics before treatment data")
            st.dataframe(well_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write("Simple Imputer (mean)")
            st.dataframe(imputed_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write(f'Rows before: {rows_before}\n\nRows after: {rows_after}')

        else:
            imputed_data = well_data.copy()
            imputer2 = KNNImputer(n_neighbors=5)
            imputed_data = pd.DataFrame(imputer2.fit_transform(imputed_data), columns=imputed_data.columns)
            rows_after = len(imputed_data)
            st.write("Stadistics before treatment data")
            st.dataframe(well_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write("KNN Imputer")
            st.dataframe(imputed_data.describe().loc[["mean", "std"]], use_container_width=True)
            st.write(f'Rows before: {rows_before}\nRows after: {rows_after}\n')
        st.success("Method applied successfuly")

        st.subheader(f"{selected_method} missingness matrix:")
        fig, ax = plt.subplots(figsize=(12,10))  # Create a figure and axes
        msno.matrix(imputed_data,ax=ax)
        st.pyplot(fig)
    
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
    
    def log_rd(df, dicNamesCurves):
        # Buscar la columna correspondiente a la resistividad profunda en el diccionario
        resistividad_deep_col = None
        for key, value in dicNamesCurves.items():
            if key == "Resistivity Deep":
                for nombre_col in value['nombre']:
                    if nombre_col in df.columns:
                        resistividad_deep_col = nombre_col
                        break
                break

        # Si se encontró la columna de resistividad profunda
        if resistividad_deep_col is not None:
            # Aplicar logaritmo a la columna correspondiente
            df[resistividad_deep_col] = np.log(df[resistividad_deep_col])
        else:
            print("No Deep Resistivity was found in the DataFrame.")

        return df
        
    imputed_data = log_rd(imputed_data,dicNamesCurves)
    imputed_data = rename_columns(imputed_data,dicNamesCurves)


    # Título de la aplicación
    st.title('LAS File Selection Model and Prediction')

    # Descripción de la aplicación
    st.write("""The following models are used to predict lithology from specific curves of
    LAS file. These are Gamma Ray (GR), Resistivity Deep (RD), Photoelectrical Factor (PEF), 
    Neutron Porosity (NPHI) and Density (RHOB). """)

    # Selección del modelo de clasificación
    selection = st.radio('Select the classification model', ('Machine Learning', 'Neuronal Network'))
    st.warning("Please make sure that the LAS file contains the input curves.")

    # Verificar que las columnas necesarias están en los datos imputados
    required_columns = ['GR', 'RD', 'PEF', 'NPHI', 'RHOB']
    if all(col in imputed_data.columns for col in required_columns):
        X = imputed_data[required_columns]
    else:
        X = pd.DataFrame(columns=required_columns)  

    # Contenedor inferior para el botón de predicción
    bottom_container = st.container()

    curves_res = required_columns  # Esto es solo un marcador de posición
    
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
                    facies_colors = ['#00008B', '#2CA25F', 'gold']
                    imputed_data['Facies_id']=y_pred
                    facies_plot.facies_plot(imputed_data,curves,facies_colors,las_file)
                    
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
                    X = scaler.fit_transform(X)
                    y_pred = loaded_rnn.predict(X)
                    y_pred = np.argmax(y_pred, axis=1)
                    st.write("Predictions:")
                    st.dataframe(y_pred)
                    curves = ['GR','NPHI','RHOB','RD']
                    facies_colors = ['#00008B', '#2CA25F', 'gold']
                    imputed_data['Facies_id']=y_pred
                    facies_plot.facies_plot(imputed_data,curves,facies_colors,las_file)

