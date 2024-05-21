def facies_plot(dataframe, curves, well_name,scale_color):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    import streamlit as st

    # Definir los colores por defecto para las curvas

    fig = make_subplots(rows=1, cols=len(curves) + 1, subplot_titles=curves + ['Litologia'], shared_yaxes=True)
    color = ['#000000','#0000FF','#FF0000','#008000','#FF00FF','gold','#00008B']

    for i, curve in enumerate(curves, start=1):
        # Seleccionar el color de la curva
        fig.add_trace(go.Scatter(y=dataframe['DEPTH'], x=dataframe[curve], mode='lines', name=curve, line=dict(color=color[i])), row=1, col=i)
        
        fig.update_xaxes(showgrid=True,gridcolor='LightGray',showline=True,mirror=True)
        fig.update_yaxes(showgrid=True,gridcolor='LightGray',showline=True,mirror=True)

    # Creamos la matriz de cluster para el Heatmap
    cluster = np.repeat(dataframe['Facies_id'].values.reshape(-1, 1), 30, axis=1)

    # Creamos una escala de color personalizada para el Heatmap
    heatmap_colors = scale_color

    # Creamos el Heatmap
    fig.add_trace(go.Heatmap(y=dataframe['DEPTH'], z=cluster, showscale=True,colorscale=[[0,'#00008B'],[0.33,'#00008B'], [0.33,'#2CA25F'],[0.66,'#2CA25F'], [0.66,'gold'],[1,'gold']], name='Facies_id',colorbar=dict(title='Litología',
                tickvals=[0,1,2],
                ticktext=['0: Caliza', '1: Lutita', '2: Arenisca'],
                x=1.2                                                                       
            )), row=1, col=len(curves) + 1)
    
    # Configuramos el diseño de la figura
    fig.update_layout(height=1000, width=1000, title_text="Well: " + well_name, yaxis=dict(title="Depth (ft)", autorange='reversed'))

    fig.update_xaxes( tickangle = 90)
    # Ocultar las etiquetas del eje x en el subplot del heatmap
    fig.update_xaxes(showticklabels=False,showgrid=False, row=1, col=len(curves) + 1)

    # Ajustar el color y la posición de los títulos de los subplots    
    for annotation in fig['layout']['annotations']:
        #annotation['font'] = dict(color='black')  # Cambiar color a negro
        annotation['y'] += 0.01


   
    # Mostramos la figura interactiva
    st.plotly_chart(fig,use_container_width=True)

   