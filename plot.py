def plot(las_file, well_data):
    import streamlit as st
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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




