def plot(las_file, well_data):
    import streamlit as st
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import plotly.graph_objs as go
    import plotly.offline as pyo

    
    st.title('LAS File Visualisation')
    
    if not las_file:
        st.warning('No file has been uploaded')
    
    else:
        
        st.write('Expand one of the following to visualise your well data.')
        st.write("""Each plot can be interacted with. To change the scales of a plot/track,         click on the left hand or right hand side of the scale and change the value as             required.""")
   
        columns = well_data.columns
        with st.expander('Histograms'):
            select_curves = st.radio('Select if you want to plot all curves or personalized curves', ('All','Select Curves'))
            multi_curve = st.multiselect('Select a Curve', columns)
            
            if select_curves == 'All':
                curves = well_data.columns
                rows = math.ceil(len(curves) / 3)
                cols = 3
                k = int(1 + 3.322 * math.log(len(well_data), 10))
                colores = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 
               'pink', 'olive', 'turquoise', 'gold', 'lavender', 'teal', 'gray', 'lightblue', 'lightgreen', 
               'lightgray', 'darkblue', 'darkgreen', 'darkred', 'darkorange', 'darkmagenta', 'darkcyan', 
               'darkslateblue', 'darkgray', 'lightcoral', 'lightpink']
                # Define the figure and dimensions
                fig = plt.figure(figsize=(15, 3 * rows))

                # Loop
                for i, feature in enumerate(curves):
                    # Add the graphic and indicate its location
                    ax = fig.add_subplot(rows, cols, i + 1)

                    # Get color cyclically from color list
                    color = colores[i % len(colores)]

                    # Create the distribution chart
                    sns.distplot(las_file[feature], bins=k, color=color, ax=ax, hist_kws=dict(edgecolor="grey", linewidth=2))

                    # Secondary configuration of the chart
                    ax.set_title(feature + " Distribution", y=1.02, size=12)  # Title
                    ax.grid()  # Enable grid
                    ax.set_axisbelow(True)  # Location grid
                    ax.set_ylabel("Density", size=12, labelpad=12)  # Y Title
                    ax.set_xlabel(feature, size=12, labelpad=12)  # X Title
                    ax.set_xlim(ax.get_xticks()[0], ax.get_xticks()[-1])  # X limit
                    ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])  # Y Limit

                # Show figure
                plt.tight_layout()
                plt.show()
                st.pyplot(fig)
                
                
            elif select_curves == 'Select Curves':
                if len(multi_curve) <= 1:
                    st.warning('Please select at least 2 curves.')
                else:
                    curves = multi_curve
                    rows = math.ceil(len(curves) / 3)
                    cols = 3
                    k = int(1 + 3.322 * math.log(len(well_data), 10))
                    colores = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 
               'pink', 'olive', 'turquoise', 'gold', 'lavender', 'teal', 'gray', 'lightblue', 'lightgreen', 
               'lightgray', 'darkblue', 'darkgreen', 'darkred', 'darkorange', 'darkmagenta', 'darkcyan', 
               'darkslateblue', 'darkgray', 'lightcoral', 'lightpink']
                    # Define the figure and dimensions
                    fig = plt.figure(figsize=(15, 3 * rows))

                    # Loop
                    for i, feature in enumerate(curves):
                        # Add the graphic and indicate its location
                        ax = fig.add_subplot(rows, cols, i + 1)

                        # Get color cyclically from color list
                        color = colores[i % len(colores)]

                        # Create the distribution chart
                        sns.distplot(las_file[feature], bins=k, color=color, ax=ax, hist_kws=dict(edgecolor="grey", linewidth=2))

                        # Secondary configuration of the chart
                        ax.set_title(feature + " Distribution", y=1.02, size=12)  # Title
                        ax.grid()  # Enable grid
                        ax.set_axisbelow(True)  # Location grid
                        ax.set_ylabel("Density", size=12, labelpad=12)  # Y Title
                        ax.set_xlabel(feature, size=12, labelpad=12)  # X Title
                        ax.set_xlim(ax.get_xticks()[0], ax.get_xticks()[-1])  # X limit
                        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])  # Y Limit

                    # Show figure
                    plt.tight_layout()
                    plt.show()
                    st.pyplot(fig)
                    
        with st.expander('Plot curves'):
            select_curves_ = st.radio('Select if you want to plot all curves or personalized curves', ('All','Select Curves'))
            multi_curve_ = st.multiselect('Select a Curve', columns)
            
            if select_curves_ == 'All':
                df_numerico = well_data.select_dtypes(include=['number'])
                num_cols = len(well_data.columns)

                # Calcula el rango intercuartílico (IQR) para cada columna
                Q1 = df_numerico.quantile(0.25)
                Q3 = df_numerico.quantile(0.75)
                IQR = Q3 - Q1

                # Define el límite para identificar outliers para cada columna
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR

                # Encuentra las columnas que tienen outliers
                cols_con_outliers = []
                
                for col in well_data.curves:
                    outliers = df_numerico[col][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                    if not outliers.empty:
                        cols_con_outliers.append(col)

                # Crea el trace para las curvas con outliers
                fig = go.Figure()
                # Verificar si 'DEPTH' o 'DEPT' está en el DataFrame
                depth_column = 'DEPTH' if 'DEPTH' in df_numerico.columns else 'DEPT' if 'DEPT' in df_numerico.columns else None

                if depth_column is None:
                    raise ValueError("Neither 'DEPTH' nor 'DEPT' column found in the DataFrame.")

                # Itera sobre las columnas con outliers y agrega la línea de la curva
                for col in cols_con_outliers:
                    # Agrega la curva al gráfico, utilizando DEPTH como eje y y la columna actual como eje x
                    fig.add_trace(go.Scatter(x=df_numerico[col], y=df_numerico[depth_column], mode='lines', name=col))

                    # Identifica los outliers y los agrega al trace
                    outliers = df_numerico[col][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                    # Ajusta las coordenadas y de los outliers para que coincidan con la columna DEPTH
                    outliers_depth = df_numerico[depth_column][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                    fig.add_trace(go.Scatter(x=outliers, y=outliers_depth, mode='markers', name=f'{col} - Outliers', 
                                            marker=dict(color='red', size=8), 
                                            hovertemplate='Valor: %{x:.2f}<extra></extra>'))

                # Configuraciones del diseño del gráfico
                fig.update_layout(title='Curvas con Outliers Marcados',
                                xaxis_title='Valor',
                                yaxis_title='DEPTH',
                                yaxis=dict(autorange="reversed"),
                                showlegend=True,
                                hovermode='closest',
                                template='plotly_white',
                                width=600,  # Ancho de la figura
                                height=900) # Alto de la figura

                # Muestra el gráfico interactivo
                st.plotychar(fig)
                    
            elif select_curves_ == 'Select Curves':
                if len(multi_curve_) <= 1:
                    st.warning('Please select at least 2 curves.')
                else:
                
                    df_numerico = well_data.select_dtypes(include=['number'])
                    num_cols = len(well_data[multi_curve_])

                    # Calcula el rango intercuartílico (IQR) para cada columna
                    Q1 = df_numerico.quantile(0.25)
                    Q3 = df_numerico.quantile(0.75)
                    IQR = Q3 - Q1

                    # Define el límite para identificar outliers para cada columna
                    limite_inferior = Q1 - 1.5 * IQR
                    limite_superior = Q3 + 1.5 * IQR

                    # Encuentra las columnas que tienen outliers
                    cols_con_outliers = []

                    for col in well_data[multi_curve_]:
                        outliers = df_numerico[col][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                        if not outliers.empty:
                            cols_con_outliers.append(col)

                    # Crea el trace para las curvas con outliers
                    fig = go.Figure()
                    # Verificar si 'DEPTH' o 'DEPT' está en el DataFrame
                    depth_column = 'DEPTH' if 'DEPTH' in df_numerico.columns else 'DEPT' if 'DEPT' in df_numerico.columns else None

                    if depth_column is None:
                        raise ValueError("Neither 'DEPTH' nor 'DEPT' column found in the DataFrame.")

                    # Itera sobre las columnas con outliers y agrega la línea de la curva
                    for col in cols_con_outliers:
                        # Agrega la curva al gráfico, utilizando DEPTH como eje y y la columna actual como eje x
                        fig.add_trace(go.Scatter(x=df_numerico[col], y=df_numerico[depth_column], mode='lines', name=col))

                        # Identifica los outliers y los agrega al trace
                        outliers = df_numerico[col][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                        # Ajusta las coordenadas y de los outliers para que coincidan con la columna DEPTH
                        outliers_depth = df_numerico[depth_column][(df_numerico[col] < limite_inferior[col]) | (df_numerico[col] > limite_superior[col])]
                        fig.add_trace(go.Scatter(x=outliers, y=outliers_depth, mode='markers', name=f'{col} - Outliers', 
                                                marker=dict(color='red', size=8), 
                                                hovertemplate='Valor: %{x:.2f}<extra></extra>'))

                    # Configuraciones del diseño del gráfico
                    fig.update_layout(title='Curvas con Outliers Marcados',
                                    xaxis_title='Valor',
                                    yaxis_title='DEPTH',
                                    yaxis=dict(autorange="reversed"),
                                    showlegend=True,
                                    hovermode='closest',
                                    template='plotly_white',
                                    width=600,  # Ancho de la figura
                                    height=900) # Alto de la figura

                    # Muestra el gráfico interactivo
                    st.ploty_chart(fig)





