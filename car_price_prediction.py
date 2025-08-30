import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Car Price Prediction", layout="centered")

# Add custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    font-weight: 700;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 1.5rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 2rem;
    font-weight: 600;
    color: #424242;
    margin-bottom: 1.5rem;
}
.info-text {
    font-size: 1.1rem;
    color: #616161;
}
.prediction-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}
.predict-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2E7D32;
    margin-bottom: 1.2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">Car Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application helps you predict the selling price of a car based on various features like year, driven kilometers, fuel type, etc.</p>', unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Dataset upload in main area
st.markdown("### Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"], help="Upload is required to use the application")

# Dataset format information
with st.expander("Dataset Format Information"):
    st.markdown("""
    Your dataset should have the following columns:
    - Car_Name: Name of the car model
    - Year: Year of manufacture
    - Selling_Price: Target variable - the price to predict
    - Present_Price: Current ex-showroom price
    - Driven_kms: Kilometers driven
    - Fuel_Type: Type of fuel (Petrol, Diesel, CNG)
    - Selling_type: Dealer or Individual
    - Transmission: Manual or Automatic
    - Owner: Number of previous owners
    """)
    
    # Show sample format
    st.markdown("**Sample CSV Format:**")
    sample_data = pd.DataFrame({
        'Car_Name': ['car1', 'car2'],
        'Year': [2018, 2016],
        'Selling_Price': [8.5, 6.75],
        'Present_Price': [10.5, 9.25],
        'Driven_kms': [5000, 15000],
        'Fuel_Type': ['Petrol', 'Diesel'],
        'Selling_type': ['Dealer', 'Individual'],
        'Transmission': ['Manual', 'Automatic'],
        'Owner': [0, 1]
    })
    st.dataframe(sample_data)
    
    # Download sample template
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample Template",
        data=csv,
        file_name="car_price_template.csv",
        mime="text/csv",
        help="Download a template CSV file with the required columns"
    )

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        
        # Validate required columns
        required_columns = ['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Driven_kms', 
                           'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.error("Please upload a valid dataset with all required columns.")
            st.stop()
        else:
            st.success("Dataset loaded successfully!")
            
            # Option to download the current dataset
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Current Dataset",
                data=csv,
                file_name="car_data.csv",
                mime="text/csv",
                help="Download the currently loaded dataset as a CSV file"
            )
            
            # Show data overview
            st.markdown("### Dataset Overview")
            st.dataframe(data.head())
            
            # Show basic statistics
            with st.expander("Dataset Statistics"):
                st.dataframe(data.describe())
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.error("Please upload a valid CSV file.")
        st.stop()
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# Only proceed with tabs if data is loaded
if 'data' in locals():
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Model Selection", "Price Prediction"])
    
    # Data Exploration tab
    with tab1:
        st.markdown('<p class="sub-header">Data Exploration</p>', unsafe_allow_html=True)
    
        # Show dataset info
        st.markdown("### Dataset Information")
        buffer = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(dtype) for dtype in data.dtypes],
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(buffer)
        
        # Visualizations
        st.markdown("### Data Visualizations")
        
        # Select visualization type
        viz_type = st.selectbox("Select Visualization", ["Price Distribution", "Year vs Price", "Fuel Type Analysis", "Transmission Analysis", "Correlation Heatmap"])
        
        if viz_type == "Price Distribution":
            st.subheader("Selling Price Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data["Selling_Price"], kde=True, ax=ax)
            ax.set_xlabel("Selling Price (in lakhs)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        elif viz_type == "Year vs Price":
            st.subheader("Car Year vs Selling Price")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x="Year", y="Selling_Price", data=data, ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Selling Price (in lakhs)")
            st.pyplot(fig)
        
        elif viz_type == "Fuel Type Analysis":
            st.subheader("Selling Price by Fuel Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Fuel_Type", y="Selling_Price", data=data, ax=ax)
            ax.set_xlabel("Fuel Type")
            ax.set_ylabel("Selling Price (in lakhs)")
            st.pyplot(fig)
        
        elif viz_type == "Transmission Analysis":
            st.subheader("Selling Price by Transmission Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Transmission", y="Selling_Price", data=data, ax=ax)
            ax.set_xlabel("Transmission Type")
            ax.set_ylabel("Selling Price (in lakhs)")
            st.pyplot(fig)
        
        elif viz_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # Model Selection tab
    with tab2:
        st.markdown('<p class="sub-header">Model Selection</p>', unsafe_allow_html=True)
    
        # Define function to train and evaluate models
        @st.cache_resource
        def train_models(model_name):
            # Data preprocessing
            X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
            y = data['Selling_Price']
            
            # Define categorical and numerical features
            categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
            numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ])
            
            # Select model based on user choice
            if model_name == "Random Forest":
                regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                regressor = LinearRegression()
            elif model_name == "Decision Tree":
                from sklearn.tree import DecisionTreeRegressor
                regressor = DecisionTreeRegressor(random_state=42)
            elif model_name == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                regressor = GradientBoostingRegressor(random_state=42)
            
            # Create pipeline with preprocessor and model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return model, mse, r2, X_test, y_test
        
        # Model selection
        st.markdown("### Select Regression Model")
        model_name = st.selectbox("Choose a regression model", 
                                 ["Random Forest", "Linear Regression", "Decision Tree", "Gradient Boosting"])
        
        # Train selected model
        if st.button("Train Model"):
            with st.spinner(f"Training {model_name} model..."):
                model, mse, r2, X_test, y_test = train_models(model_name)
                
                # Display model performance
                st.markdown("### Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Store model in session state for use in prediction tab
                st.session_state['trained_model'] = model
                st.session_state['model_name'] = model_name
                st.success(f"{model_name} model trained successfully! Go to the Price Prediction tab to make predictions.")
                
                # Feature importance (if applicable)
                if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                    st.markdown("### Feature Importance")
                    
                    # Get feature names after one-hot encoding
                    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
                    numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']
                
                    # Get the feature importance from the model
                    feature_importance = model.named_steps['regressor'].feature_importances_
                    
                    # Get the feature names after preprocessing
                    ohe = model.named_steps['preprocessor'].transformers_[1][1]
                    cat_feature_names = ohe.get_feature_names_out(categorical_features)
                    feature_names = numerical_features + list(cat_feature_names)
                    
                    # Create a dataframe for feature importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)

    # Price Prediction tab
    with tab3:
        st.markdown('<p class="sub-header">Car Price Prediction</p>', unsafe_allow_html=True)
        
        # Check if model is trained
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model in the Model Selection tab first.")
        else:
            model = st.session_state['trained_model']
            model_name = st.session_state['model_name']
            
            st.info(f"Using trained {model_name} model for predictions.")
            
            # Input form for prediction
            st.markdown('<p class="predict-title">Predict Car Price</p>', unsafe_allow_html=True)
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input("Year of Purchase", min_value=2000, max_value=2023, value=2015)
                present_price = st.number_input("Present Price (in lakhs)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
                driven_kms = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=25000, step=1000)
                owner = st.number_input("Number of Previous Owners", min_value=0, max_value=3, value=0)
            
            with col2:
                fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
                selling_type = st.selectbox("Selling Type", options=["Dealer", "Individual"])
                transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Make prediction
            if st.button("Predict Price"):
                # Create a dataframe with the input values
                input_data = pd.DataFrame({
                    'Year': [year],
                    'Present_Price': [present_price],
                    'Driven_kms': [driven_kms],
                    'Fuel_Type': [fuel_type],
                    'Selling_type': [selling_type],
                    'Transmission': [transmission],
                    'Owner': [owner]
                })
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.success(f"### Predicted Selling Price: ₹ {prediction:.2f} Lakhs")