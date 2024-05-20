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
    st.write('**Curve Information**')
    for count, curve in enumerate(las_file.curves):
        st.write(f"Curve: {curve.mnemonic}, \t\t Units: {curve.unit}, \t Description: {curve.descr}")
    st.write(f"<b>There are a total of: {count+1} curves present within this file</b>", unsafe_allow_html=True)

    # Curve Statistics
    st.write('<b>Curve Statistics</b>', unsafe_allow_html=True)
    st.write(well_data.describe())
    
    
        
