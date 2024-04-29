import streamlit as st 
import PIL as pil 
import datetime as dt

st.header('Web-сервис на основе python (streamlit)')

if "messages" not in st.session_state:
    st.text('********************************************')
    st.text("st.session_state: "+str(st.session_state))
    st.text('********************************************')
    st.text(dt.time())    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:   
        st.text(message)
            
    