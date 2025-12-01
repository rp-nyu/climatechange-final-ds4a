import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import figure_factory
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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




# PREDICTION PAGE
elif app_mode == "Prediction":
    st.title("Climate Prediction: Model Comparison 🌡️")
    st.markdown("Compare the results of a **Linear Regression** (simple, interpretable) and **Random Forest** (complex, non-linear) model on predicting **Average Temperature** for the selected country.")

    # --- 1. SELECTION ---
    st.header("Select Country and Model")
    
    # Country Selector
    selected_country_pred = st.selectbox(
        "Select Country to Analyze", df['Country'].unique(), index=0
    )
    
    # NEW: Model Selector
    model_choice = st.selectbox(
        "Select Regression Model",
        ["Random Forest Regressor", "Linear Regression"],
        index=0 # Default to Random Forest
    )
    
    # Target is fixed to Avg Temperature
    target_variable = 'Avg Temperature (°C)'
    st.write(f"**Target:** {target_variable}")

    # Define all available predictor features (excluding the target and Country)
    predictor_variables = [
        'Year', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
        'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]
    
    st.info(f"The model will use **{len(predictor_variables)} features** to predict temperature trends for **{selected_country_pred}**.")
    st.markdown("---")


    # --- 2. DATA PREPARATION (COUNTRY-SPECIFIC AND AGGREGATED) ---
    st.header(f"Model Training for {selected_country_pred}")
    
    # 1. FILTER DATA BY SELECTED COUNTRY
    country_df = df[df['Country'] == selected_country_pred].copy()
    
    # 2. AGGREGATION FIX: Group by Year and calculate the mean for all numeric columns.
    # This is CRUCIAL to remove the duplicate year entries that were messing up the plot.
    agg_cols = [col for col in predictor_variables if col != 'Year'] + [target_variable]
    
    # Calculate the mean for all features for each year
    clean_country_df = country_df.groupby('Year')[agg_cols].mean().reset_index()

    # 3. Drop any rows with remaining missing values after aggregation
    model_df = clean_country_df.dropna()
    
    if len(model_df) < 10:
        st.error(f"Not enough clean data points for {selected_country_pred} (only {len(model_df)} years of data). Please select a country with more complete data.")
        st.stop()
        
    X = model_df[predictor_variables]
    y = model_df[target_variable]
    
    # Train Model based on user choice
    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    else: # Linear Regression
        model = LinearRegression()
        
    model.fit(X, y)
    y_pred_full = model.predict(X) 
    
    # Evaluation Metrics
    r2 = r2_score(y, y_pred_full)
    
    col1, col2 = st.columns(2)
    col1.metric("Selected Model", model_choice)
    col2.metric("R² Score (Model Fit)", f"**{r2:.4f}**")
    
    if r2 > 0.8:
         st.success("R² Status: Very Strong Fit! 🚀")
    elif r2 > 0.6:
         st.info("R² Status: Good Fit. 👍")
    else:
         st.warning("R² Status: Weak Fit. 📉")
    
    st.markdown("---")


    # --------------------------------------------------------------------------------------------------
    # --- 3. ACTUAL VS PREDICTED TREND (CLEAN VISUALIZATION) ---
    # --------------------------------------------------------------------------------------------------
    st.header("Actual vs. Predicted Trend Over Time")
    st.info("The actual temperature data has been averaged by year to ensure a smooth, chronological trendline.")
    
    # Create a DataFrame for plotting, combining Actual and Predicted data (already sorted by Year due to groupby)
    plot_df = pd.DataFrame({
        'Year': model_df['Year'],
        'Actual Temperature': y,
        'Predicted Temperature': y_pred_full
    })
    
    # Create the Plotly chart
    fig_trend = go.Figure()

    # Trace 1: Actual Temperature (Blue line)
    fig_trend.add_trace(go.Scatter(
        x=plot_df['Year'], y=plot_df['Actual Temperature'], 
        mode='lines+markers', name='Actual Temperature',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Trace 2: Predicted Temperature (Red line)
    fig_trend.add_trace(go.Scatter(
        x=plot_df['Year'], y=plot_df['Predicted Temperature'], 
        mode='lines+markers', name='Predicted Temperature',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=6)
    ))

    fig_trend.update_layout(
        title=f"Actual vs. Predicted Temperature Trend for {selected_country_pred} ({model_choice})",
        xaxis_title="Year",
        yaxis_title="Average Temperature (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")


    # --- 4. MODEL INSIGHTS ---
    st.header("Model Insights")
    
    if model_choice == "Random Forest Regressor":
        st.subheader("Feature Importance (Random Forest)")
        st.info("This shows the relative power of each factor in predicting temperature.")
        
        # Get feature importances from the trained RF model
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        # Plot the feature importance
        fig_imp = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance for Temperature in {selected_country_pred}"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig_imp, use_container_width=True)
        
    else: # Linear Regression
        st.subheader("Model Coefficients (Linear Regression)")
        st.info("This shows the direction and magnitude of the straight-line relationship for each factor.")

        # Display the coefficients
        coef_df = pd.DataFrame({
            'Feature': predictor_variables,
            'Coefficient': model.coef_.round(6)
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        
        st.dataframe(coef_df, use_container_width=True)
        
        st.markdown(f"""
        **Interpretation:** A positive coefficient means an increase in that feature drives temperature **up**.
        Baseline Temperature Estimate (Intercept): **{model.intercept_:.4f} °C**
        """)

    st.markdown("---")


    # --- 5. INTERACTIVE PREDICTOR TOOL ---
    st.header("Interactive Prediction Tool")
    st.info(f"Adjust the factors below to see the predicted **{target_variable}**.")
    
    # Create columns for inputs (4 columns per row)
    input_cols = st.columns(min(4, len(predictor_variables)))
    input_data = {}
    
    for i, feature in enumerate(predictor_variables):
        with input_cols[i % len(input_cols)]:
            min_val = X[feature].min()
            max_val = X[feature].max()
            
            default_val = X[feature].mean()

            user_input = st.number_input(
                f"Input **{feature}**:",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                key=f"input_{feature}"
            )
            input_data[feature] = user_input

    # Prepare input for prediction
    input_df = pd.DataFrame([input_data])
    predicted_target = model.predict(input_df)[0]

    st.success(f"**Predicted Avg Temperature:** **{predicted_target:.2f}** °C")

    st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")