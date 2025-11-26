import streamlit as st

data = "climate_change_dataset.csv"

st.set_page_config(
    page_title="Linear Regression App ", layout="wide", page_icon="images/climage-change-icon.png"
)

st.sidebar.title("Pages:")
selected_page = st.sidebar.radio("Select Page", ["Introduction", "Visualization", "Prediction", "Conclusion"])

st.title("Climate Change Prediction 🔥")

if selected_page == "Introduction": 
    st.image("images/drid-polar-bear.jpg", use_container_width=True)
    st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")

    st.markdown("### Introduction")
