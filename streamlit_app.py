import streamlit as st


st.sidebar.title("Pages:")
selected_page = st.sidebar.radio("Select Page", ["Introduction", "Visualization", "Prediction", "Conclusion"])

st.title("Climate Change Prediction")

if selected_page == "Introduction": 
    st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")
    st.image("drid-polar-bear.jpg", use_container_width=True)

    st.markdown("### Introduction")
