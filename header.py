def header(las_file):
    import streamlit as st
    st.title('LAS File Header Info')
    if not las_file:
        st.warning('No file has been uploaded')
    else:
        for item in las_file.well:
            st.write(f"<b>{item.descr.capitalize()} ({item.mnemonic}):</b> {item.value}",                 unsafe_allow_html=True)
    
    
