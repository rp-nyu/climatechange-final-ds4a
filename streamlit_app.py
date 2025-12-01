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
import matplotlib.pyplot as plt
import shap


df = pd.read_csv("climate_change_dataset.csv")

# --- Hide Streamlit header and footer ---
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

# --- Sidebar Navigation ---
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',
                                ['Introduction','Visualization','Prediction', 
                                 "AI Explainability", "Hyperparameter Tuning"])

st.title("Climate Change Prediction 🔥")

# ===========================
# INTRODUCTION PAGE
# ===========================
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
    st.markdown("Missing values are known as null or NaN values. Missing data tends to introduce bias that leads to misleading results.")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:", totalmiss)

    if totalmiss <= 30:
        st.success("Looks good! We have less than 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")

    st.markdown("### Completeness")
    st.markdown("Completeness is defined as the ratio of non-missing values to total records in the dataset.") 
    nonmissing = (df.notnull().sum().round(2))
    completeness = round(sum(nonmissing)/len(df), 2)
    st.write("Completeness ratio:", completeness)

    if completeness >= 0.80:
        st.success("Looks good! We have a completeness ratio greater than 0.85.")
    else:
        st.success("Poor data quality due to low completeness ratio (less than 0.85).")

# ===========================
# VISUALIZATION PAGE
# ===========================
elif app_mode == "Visualization":
    st.title("Climate Trends: Visual Exploration 📈")

    # --- Country Selector ---
    selected_country = st.selectbox("Select a country", df['Country'].unique())

    # --- Variable Selector ---
    numeric_cols = [
        'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)', 
        'Sea Level Rise (mm)', 'Rainfall (mm)',
        'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]
    selected_variable = st.selectbox("Select variable to plot", numeric_cols)

    # --- Plot Type Selector ---
    plot_types = ["Line with mean ± std", "Scatter & trend", "Box plot", "Violin plot"]
    selected_plot_type = st.selectbox("Select plot type", plot_types)

    # --- Filter Data ---
    country_data = df[df['Country'] == selected_country]

    fig = None

    # --- Line Plot with Std Shading ---
    if selected_plot_type == "Line with mean ± std":
        agg = country_data.groupby('Year')[selected_variable].agg(['mean', 'std']).reset_index()

        x = agg['Year']
        mean = agg['mean'].interpolate()
        std = agg['std'].interpolate()

        fig = go.Figure()

        # Std shaded region
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),
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

    # --- Scatter with Trendline ---
    elif selected_plot_type == "Scatter & trend":
        fig = px.scatter(
            country_data,
            x='Year',
            y=selected_variable,
            trendline='ols',
            title=f"{selected_variable} per year for {selected_country}"
        )

    # --- Box Plot ---
    elif selected_plot_type == "Box plot":
        fig = px.box(
            country_data,
            x='Year',
            y=selected_variable,
            points='all',
            title=f"{selected_variable} distribution per year for {selected_country}"
        )

    # --- Violin Plot ---
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

# ===========================
# PREDICTION PAGE
# ===========================
elif app_mode == "Prediction":
    st.title("Climate Prediction: Model Comparison 🌡️")
    st.markdown("Compare Linear Regression and Random Forest models on predicting **Average Temperature**.")

    # --- Country and Model Selection ---
    st.header("Select Country and Model")
    selected_country_pred = st.selectbox("Select Country to Analyze", df['Country'].unique(), index=0)

    model_choice = st.selectbox("Select Regression Model",
                                ["Random Forest Regressor", "Linear Regression"],
                                index=0)

    target_variable = 'Avg Temperature (°C)'
    st.write(f"**Target:** {target_variable}")

    predictor_variables = [
        'Year', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
        'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]
    
    st.markdown("---")

    # --- Prepare Data ---
    st.header(f"Model Training for {selected_country_pred}")

    country_df = df[df['Country'] == selected_country_pred].copy()

    agg_cols = [col for col in predictor_variables if col != 'Year'] + [target_variable]

    clean_country_df = country_df.groupby('Year')[agg_cols].mean().reset_index()

    model_df = clean_country_df.dropna()
    
    if len(model_df) < 10:
        st.error(f"Not enough clean data for {selected_country_pred}.")
        st.stop()

    X = model_df[predictor_variables]
    y = model_df[target_variable]

    # --- Train Model ---
    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    else:
        model = LinearRegression()
        
    model.fit(X, y)
    y_pred_full = model.predict(X)

    r2 = r2_score(y, y_pred_full)

    col1, col2 = st.columns(2)
    col1.metric("Selected Model", model_choice)
    col2.metric("R² Score (Model Fit)", f"{r2:.4f}")

    if r2 > 0.8:
         st.success("R² Status: Very Strong Fit! 🚀")
    elif r2 > 0.6:
         st.info("R² Status: Good Fit 👍")
    else:
         st.warning("R² Status: Weak Fit 📉")
    
    st.markdown("---")

    # --- Actual vs Predicted Plot ---
    st.header("Actual vs. Predicted Trend Over Time")
    st.info("Actual temperature values were averaged by year for consistency.")

    plot_df = pd.DataFrame({
        'Year': model_df['Year'],
        'Actual Temperature': y,
        'Predicted Temperature': y_pred_full
    })
    
    fig_trend = go.Figure()

    # Actual
    fig_trend.add_trace(go.Scatter(
        x=plot_df['Year'], y=plot_df['Actual Temperature'],
        mode='lines+markers', name='Actual Temperature',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ))

    # Predicted
    fig_trend.add_trace(go.Scatter(
        x=plot_df['Year'], y=plot_df['Predicted Temperature'],
        mode='lines+markers', name='Predicted Temperature',
        line=dict(color='red', width=2, dash='dot'), marker=dict(size=6)
    ))

    fig_trend.update_layout(
        title=f"Actual vs. Predicted Temperature Trend for {selected_country_pred}",
        xaxis_title="Year",
        yaxis_title="Average Temperature (°C)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")

    # --- Model Insights ---
    st.header("Model Insights")

    # Feature Importance (RF)
    if model_choice == "Random Forest Regressor":
        st.subheader("Feature Importance (Random Forest)")
        st.info("Shows the relative influence of each feature.")

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance for Temperature in {selected_country_pred}"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig_imp, use_container_width=True)
        
    else:
        # Coefficients (Linear Regression)
        st.subheader("Model Coefficients (Linear Regression)")
        st.info("Indicates the direction and strength of each feature's linear effect.")

        coef_df = pd.DataFrame({
            'Feature': predictor_variables,
            'Coefficient': model.coef_.round(6)
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        
        st.dataframe(coef_df, use_container_width=True)
        
        st.markdown(f"""
        **Interpretation:** Positive coefficients increase predicted temperature.  
        Intercept: **{model.intercept_:.4f} °C**
        """)

        st.markdown("---")

        # --- Interactive Predictor Tool ---
        st.header("Interactive Prediction Tool")
        st.info("Modify feature values to generate a temperature prediction.")

        input_cols = st.columns(min(4, len(predictor_variables)))
        input_data = {}
        
        for i, feature in enumerate(predictor_variables):
            with input_cols[i % len(input_cols)]:
                min_val = X[feature].min()
                max_val = X[feature].max()
                default_val = X[feature].mean()

                user_input = st.number_input(
                    f"Input {feature}:",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    key=f"input_{feature}"
                )
                input_data[feature] = user_input

        input_df = pd.DataFrame([input_data])
        predicted_target = model.predict(input_df)[0]

        st.success(f"Predicted Avg Temperature: **{predicted_target:.2f} °C**")

# ===========================
# AI EXPLAINABILITY PAGE
# ===========================
elif app_mode == "AI Explainability":
    import shap
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree, DecisionTreeRegressor

    st.title("🤖 Why did the model predict that?")
    st.info("Using SHAP (SHapley Additive exPlanations) to understand feature drivers.")

    # --- 1. Data Prep (Same as Prediction Page) ---
    st.sidebar.markdown("---")
    country_choice = st.sidebar.selectbox("Select Country for Analysis", df['Country'].unique())
    
    # Filter and prepare data
    country_df = df[df['Country'] == country_choice].copy()
    
    # Define features and target
    target_variable = 'Avg Temperature (°C)'
    predictor_variables = [
        'Year', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
        'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]
    
    # Clean data
    agg_cols = [col for col in predictor_variables if col != 'Year'] + [target_variable]
    clean_country_df = country_df.groupby('Year')[agg_cols].mean().reset_index()
    model_df = clean_country_df.dropna()
    
    X = model_df[predictor_variables]
    y = model_df[target_variable]

    # --- 2. Session State Management ---
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    if 'model_choice_xai' not in st.session_state:
        st.session_state.model_choice_xai = None

    # --- 3. Model Selection ---
    model_choice = st.selectbox("Choose Model to Explain", ["Random Forest", "Linear Regression"])

    if st.button("Generate Explanation"):
        with st.spinner("Calculating SHAP values..."):
            
            # Train the model freshly for this specific country data
            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                model.fit(X, y)
                # Tree Explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                model = LinearRegression()
                model.fit(X, y)
                # Linear Explainer (needs background data, we use X)
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)

            # Store in session state
            st.session_state.shap_values = shap_values
            st.session_state.model_choice_xai = model_choice
            st.session_state.X_data = X # Store X so we can plot it later

    # --- 4. Visualizations ---
    if st.session_state.shap_values is not None:
        
        # Retrieve data
        shap_values = st.session_state.shap_values
        X_display = st.session_state.X_data
        current_model = st.session_state.model_choice_xai

        st.markdown("---")

        # PLOT 1: Feature Importance (Summary Plot)
        st.subheader(f"1. Global Feature Importance ({current_model})")
        st.write("Which features impact the Temperature the most?")
        
        # Create matplotlib figure
        fig_shap1 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
        st.pyplot(fig_shap1)
        
        st.markdown("---")

        # PLOT 2: Dependence Plot
        st.subheader("2. Deep Dive: Feature Dependence")
        st.write("How does a specific feature value change the temperature prediction?")
        
        selected_feature = st.selectbox("Select feature to analyze:", predictor_variables, index=0)
        
        if selected_feature in X_display.columns:
            fig_shap2, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(selected_feature, shap_values, X_display, ax=ax, show=False)
            st.pyplot(fig_shap2)
        else:
            st.warning(f"Could not find {selected_feature} in dataset.")

        st.markdown("---")

        # PLOT 3: Tree Visualization (Only for Random Forest)
        st.subheader("3. Random Forest: Under the Hood")
        
        if current_model == "Random Forest":
            st.write("Below is **one single tree** from the forest to demonstrate the logic.")
            
            # Train a small tree just for visualization
            viz_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
            viz_tree.fit(X, y)
            
            # Plot Tree
            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
            plot_tree(viz_tree, feature_names=X.columns, filled=True, rounded=True, impurity=False, fontsize=10, ax=ax_tree)
            st.pyplot(fig_tree)
        else:
            st.info("Select 'Random Forest' in the dropdown above to see the Decision Tree visualization.")

    

# ===========================
# HYPERPARAMETER TUNING PAGE
# ===========================
elif app_mode == "Hyperparameter Tuning":
    st.title("Model Optimization: Hyperparameter Tuning ⚙️📊")


# ===========================
# FOOTER
# ===========================
st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")