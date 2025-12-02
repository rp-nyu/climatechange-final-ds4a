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

    st.title("🌍 AI Explainability: Understanding Climate Drivers")
    st.write("""
    This page explains *why* our models predict certain temperatures using  
    **SHAP (SHapley Additive exPlanations)** — the most widely trusted method  
    for interpreting machine-learning predictions.

    We focus on:
    - **Global feature importance** — which climate indicators matter most overall  
    - **Dependence analysis** — how each variable affects predicted temperature  
    """)

    st.markdown("---")

    # ===============================================================
    # 1. COUNTRY SELECTION
    # ===============================================================
    st.subheader("1. Choose a Country for Explainability")
    country_choice = st.selectbox("Select Country", df['Country'].unique())

    country_df = df[df['Country'] == country_choice].copy()

    target_variable = 'Avg Temperature (°C)'
    predictor_variables = [
        'Year', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)',
        'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]

    # Aggregate by year
    agg_cols = [col for col in predictor_variables if col != 'Year'] + [target_variable]
    clean_country_df = country_df.groupby('Year')[agg_cols].mean().reset_index().dropna()

    X = clean_country_df[predictor_variables]
    y = clean_country_df[target_variable]

    if len(X) < 8:
        st.warning("Not enough clean yearly data for SHAP analysis.")
        st.stop()

    st.markdown("---")

    # ===============================================================
    # 2. MODEL SELECTION
    # ===============================================================
    st.subheader("2. Choose a Model to Explain")
    model_choice = st.selectbox("Select Model", ["Random Forest Regressor", "Linear Regression"])

    # Train selected model
    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X, y)

    # ===============================================================
    # 3. GENERATE SHAP VALUES
    # ===============================================================
    if st.button("Generate SHAP Explanations 🚀"):
        with st.spinner("Computing SHAP values for climate predictors..."):

            if model_choice == "Random Forest Regressor":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)

            # SAVE to session state
            st.session_state["explainer"] = explainer
            st.session_state["shap_values"] = shap_values
            st.session_state["X_display"] = X
            st.session_state["model_used"] = model_choice

    # ===============================================================
    # 4. DISPLAY GLOBAL EXPLANATIONS
    # ===============================================================
    if "shap_values" in st.session_state:

        shap_values = st.session_state["shap_values"]
        X_display = st.session_state["X_display"]
        model_used = st.session_state["model_used"]

        st.markdown("---")
        st.subheader("3. Global Feature Importance")
        st.write("""
        This chart shows which climate indicators influence predicted temperature  
        **the most on average**.
        """)

        fig1 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
        st.pyplot(fig1)

        st.info("""
        **How to read this:**
        - Long bars = strong climate influence  
        - Positive SHAP = pushes predicted temperature up  
        - Negative SHAP = pushes predicted temperature down  
        """)

        st.markdown("---")

        # ===============================================================
        # 5. DEPENDENCE (FEATURE EFFECT ANALYSIS)
        # ===============================================================
        st.subheader("4. How a Selected Variable Affects Temperature")
        st.write("""
        This plot shows how **changes in a single climate variable**  
        (e.g., CO₂ emissions or Renewable Energy %)  
        influence predicted temperatures.
        """)

        selected_feature = st.selectbox("Select a variable to analyze:", predictor_variables)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(selected_feature, shap_values, X_display, ax=ax2, show=False)
        st.pyplot(fig2)

        st.info("""
        **Interpretation Examples:**
        - Increasing CO₂ emissions typically increases predicted temperature  
        - More renewable energy (%) often reduces predicted temperature  
        - Greater forest area (%) is associated with cooling effects  
        - Sea level rise corresponds to warming climate patterns  
        """)

# ===========================
# HYPERPARAMETER TUNING PAGE
# ===========================
elif app_mode == "Hyperparameter Tuning":
    import wandb
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import time

    st.title("Model Optimization: Hyperparameter Tuning ⚙️📊")
    st.write("""
    In this section, we fine-tune the **Random Forest Regressor** to find the best model for predicting  
    **Average Temperature (°C)** using climate indicators.
    
    We track experiments using **Weights & Biases (W&B)** to visualize results, compare runs,  
    and select the best-performing model.
    """)

    st.markdown("---")
    st.subheader("Step 1: Connect to Weights & Biases")
    st.write("Enter your W&B API key if you want to log experiments online. Otherwise, the app will run offline.")

    wb_api = st.text_input("W&B API Key (Optional)", type="password")

    if st.button("Validate API Key"):
        if wb_api:
            try:
                wandb.login(key=wb_api)
                st.success("Logged into Weights & Biases successfully!")
            except:
                st.error("Invalid API key. Running in offline mode instead.")
        else:
            st.info("No key provided — running in offline mode.")

    st.markdown("---")
    st.subheader("Step 2: Select Country for Hyperparameter Tuning")

    tuning_country = st.selectbox("Choose Country", df['Country'].unique())

    # --- Prepare data for tuning ---
    country_df = df[df['Country'] == tuning_country].copy()

    target = 'Avg Temperature (°C)'
    features = [
        'Year', 'CO2 Emissions (Tons/Capita)', 'Sea Level Rise (mm)', 
        'Rainfall (mm)', 'Population', 'Renewable Energy (%)',
        'Extreme Weather Events', 'Forest Area (%)'
    ]

    agg_cols = [col for col in features if col != 'Year'] + [target]
    model_df = country_df.groupby('Year')[agg_cols].mean().reset_index().dropna()

    if len(model_df) < 10:
        st.error("Not enough clean data for this country to run hyperparameter tuning.")
        st.stop()

    X = model_df[features]
    y = model_df[target]

    st.markdown("---")
    st.subheader("Step 3: Run Hyperparameter Search")

    st.write("We will test combinations of:")
    st.code("""
n_estimators = [20, 50, 100]
max_depth = [2, 4, 6]
    """, language="python")

    if st.button("Start Grid Search 🚀"):
        st.info("Running Grid Search for Random Forest...")

        n_estimators_list = [20, 50, 100]
        max_depth_list = [2, 4, 6]

        results = []
        total_runs = len(n_estimators_list) * len(max_depth_list)
        progress = st.progress(0)
        run_count = 0

        for n_est in n_estimators_list:
            for depth in max_depth_list:
                run_count += 1
                progress.progress(run_count / total_runs)

                # Initialize W&B run
                if wb_api:
                    wandb_run = wandb.init(
                        project="climate-hyperparameter-tuning",
                        config={"n_estimators": n_est, "max_depth": depth},
                        reinit=True
                    )
                else:
                    wandb_run = wandb.init(
                        project="climate-hyperparameter-tuning",
                        mode="disabled",  # offline
                        config={"n_estimators": n_est, "max_depth": depth},
                        reinit=True
                    )

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                # Save results
                results.append({
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "RMSE": rmse
                })

                # Log to W&B
                wandb.log({"RMSE": rmse})
                wandb_run.finish()
                time.sleep(0.1)

        st.success("Grid Search Completed!")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### Results Table")
            st.dataframe(results_df.style.highlight_min(subset=["RMSE"], color="lightgreen"))

        with col2:
            st.write("### Optimization Heatmap")
            pivot = results_df.pivot(index="max_depth", columns="n_estimators", values="RMSE")

            fig = px.imshow(
                pivot,
                text_auto=".2f",
                color_continuous_scale="viridis",
                labels=dict(color="RMSE"),
                title="RMSE Heatmap (Lower is Better)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Tuning Insights")
        st.write("""
- **Lower RMSE = better performance.**  
- Shallow trees (low max_depth) often generalize better.  
- More trees (higher n_estimators) help stabilize predictions.  
- Look for the darkest square on the heatmap — that is your best model.
        """)


# ===========================
# FOOTER
# ===========================
st.link_button("Github Repo", "https://github.com/rp-nyu/climatechange-final-ds4a")