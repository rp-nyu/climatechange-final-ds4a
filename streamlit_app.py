import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import figure_factory
import plotly.graph_objects as go


df = pd.read_csv("climate_change_dataset.csv")

# Hide the Streamlit header and footer
def hide_header_footer():
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_header_footer()

st.set_page_config(
    page_title="Climate Change Lab App ", layout="wide", page_icon="images/climage-change-icon.png"
)

# navigation dropdown
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])

## select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Wine Quality","Real Estate"])
## if select_dataset == "Wine Quality":
   ## df = pd.read_csv("wine_quality_red.csv")
## else: 
   ## df = pd.read_csv("real_estate.csv")

st.title("Climate Change Prediction 🔥")

# INTRODUCTION PAGE
if app_mode == "Introduction": 
    st.image("images/drid-polar-bear.jpg", use_container_width=True)

    st.markdown("### Introduction")
    st.write("Climate change represents one of the most significant risks to global economic stability and environmental sustainability. This dashboard serves as a strategic tool for stakeholders - governments, environmental agencies, and corporations - to monitor critical climate indicators, assess risks, and track the effectiveness of sustainability initiatives.")

    st.markdown("### Objectives")
    st.write("""
    - Identify long-term patterns in climate data (2000 - 2024) and visualize trends.
    - Understand the relationship between Renewable Energy adoption and CO2 reduction.
    - Pinpoint countries or regions with high vulnerability (e.g., high Sea Level Rise).
    """)

    st.markdown("### Dataset")
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    col1.markdown(" **Year** ")
    col1.markdown("The year in which the data was recorded (2000–2024).")

    col2.markdown(" **Country** ")
    col2.markdown("The country or region for which the climate data is recorded.")

    col3.markdown(" **Avg Temperature (°C)** ")
    col3.markdown("Average annual temperature in degrees Celsius for the country.")

    col4.markdown(" **CO2 Emissions (Tons/Capita)** ")
    col4.markdown("CO₂ emissions per person measured in metric tons.")

    col5.markdown(" **Sea Level Rise (mm)** ")
    col5.markdown("Annual sea-level rise in millimeters for coastal regions.")

    col6.markdown(" **Rainfall (mm)** ")
    col6.markdown("Total annual rainfall measured in millimeters.")

    col7.markdown(" **Population** ")
    col7.markdown("Population of the country in that year.")

    col8.markdown(" **Renewable Energy (%)** ")
    col8.markdown("Percent of total energy consumption from renewables (e.g., solar, wind).")

    col9.markdown(" **Extreme Weather Events** ")
    col9.markdown("Number of extreme events (floods, storms, wildfires) reported that year.")

    col10.markdown(" **Forest Area (%)** ")
    col10.markdown("Percent of the country's land area covered by forests.")
        
    st.markdown("A preview of the dataset is shown below:")    
    st.dataframe(df.head())
    st.write("Source: https://www.kaggle.com/datasets/bhadramohit/climate-change-dataset")

    st.markdown("### Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    if totalmiss <= 30:
        st.success("Looks good! We have less than 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")

    st.markdown("### Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    if completeness >= 0.80:
        st.success("Looks good! We have a completeness ratio greater than 0.85.")
           
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

# VISUALIZATION PAGE
elif app_mode == "Visualization":
    st.markdown("### Country-wise Climate Trends")

    # Country selector
    selected_country = st.selectbox("Select a country", df['Country'].unique())

    # Variable selector
    numeric_cols = [
        'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 
        'Sea Level Rise (mm)', 'Rainfall (mm)',
        'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]
    selected_variable = st.selectbox("Select variable to plot", numeric_cols)

    # Plot type selector
    plot_types = ["Line with mean ± std", "Scatter & trend", "Box plot", "Violin plot"]
    selected_plot_type = st.selectbox("Select plot type", plot_types)

    # Filter data for selected country
    country_data = df[df['Country'] == selected_country]

    fig = None

    if selected_plot_type == "Line with mean ± std":
        agg = country_data.groupby('Year')[selected_variable].agg(['mean', 'std']).reset_index()

        x = agg['Year']
        mean = agg['mean'].interpolate()
        std = agg['std'].interpolate()

        fig = go.Figure()

        # Shaded area for mean ± std
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),  # forward + backward for closed shape
            y=pd.concat([mean + std, (mean - std)[::-1]]),
            fill='toself',
            fillcolor='rgba(173,216,230,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='±1 Std'
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x,
            y=mean,
            mode='lines+markers',
            line=dict(color='blue'),
            name='Mean'
        ))

        fig.update_layout(
            title=f"{selected_variable} over time for {selected_country}",
            xaxis_title='Year',
            yaxis_title=selected_variable
        )

    elif selected_plot_type == "Scatter & trend":
        fig = px.scatter(
            country_data,
            x='Year',
            y=selected_variable,
            trendline='ols',
            title=f"{selected_variable} per year for {selected_country}"
        )

    elif selected_plot_type == "Box plot":
        fig = px.box(
            country_data,
            x='Year',
            y=selected_variable,
            points='all',
            title=f"{selected_variable} distribution per year for {selected_country}"
        )

    elif selected_plot_type == "Violin plot":
        fig = px.violin(
            country_data,
            x='Year',
            y=selected_variable,
            box=True,
            points='all',
            title=f"{selected_variable} distribution per year for {selected_country}"
        )


    if fig:
        st.plotly_chart(fig, use_container_width=True)


st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")

