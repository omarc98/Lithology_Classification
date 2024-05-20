def prediction_model(las_file, well_data):
    import streamlit as st
    import numpy as np
    import joblib
    
    
    st.title('LAS File Selection Model and Prediction')
    
    if not las_file:
        st.warning('No file has been uploaded')
    
    else:
        st.write("""The following models are used to predict lithology from specific curves of
        las file. 
        These are Gamma Ray (GR), Resistivity Deep(RD),Photoelectrical Factor (PEFZ),  Neutron(TNPH) and Density(RHOZ) .
        Please select these specific curves to predict.""")

        # Need to setup an empty list for len check to work
        curves = []
        columns = well_data.columns.tolist()

        col1_md, col2_md= st.columns(2)

        selection = col1_md.radio('Select the classification model', ('XGB', 'Neuronal Network'))
        
        

        if selection == 'XGB':
            curves = st.multiselect('Select Curves to Predict',columns)
        else:
            curves = st.multiselect('Select Curves To Predict', columns)

        if (len(curves) > 5) | (len(curves)<5):
            st.warning('Please select the curves GR, RD, PEFZ, TNPH, RHOZ')
        else:
           
            well_data['Log RLA5'] = np.where(well_data['RLA5'].notna() & (well_data['RLA5'] > 0),   np.log(well_data['RLA5']), well_data['RLA5'])
            target_string = "RLA5"
            replacement_string = "Log RLA5"
            curves = [replacement_string if item == target_string else item for item in curves]
            X = well_data[curves]
            
           
            loaded_model = joblib.load('classification_model.pkl')
            lithology_preds = loaded_model.predict(X)
            st.write(lithology_preds)
            
            well_data['Facies_id'] = lithology_preds
            
            st.write(well_data)
            return well_data
            

