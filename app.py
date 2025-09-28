# %%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and define expected features
model = None
model_load_error = None
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    # Don't raise on import — show a friendly error in the UI instead
    model = None
    model_load_error = str(e)
expected_features = [
    'age', 'workclass', 'fnlwgt', 'educational-num', 
    'marital-status', 'occupation', 'relationship',
    'race', 'gender', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country'
]

# Feature importance data
# Build a feature importance DataFrame only if the model exposes the attribute
if model is not None and hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': expected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    # Fallback: zero importance so the app still loads
    feature_importance = pd.DataFrame({
        'Feature': expected_features,
        'Importance': [0.0] * len(expected_features)
    }).sort_values('Importance', ascending=False)

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Add these new inputs to sidebar
workclass = st.sidebar.selectbox("Work Class", [
    "Private", "Self-emp", "Gov", "Others"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married", "Single", "Divorced", "Separated"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

def encode_categorical_features(df):
    categorical_columns = ['workclass', 'marital-status', 'occupation', 
                         'relationship', 'race', 'gender', 'native-country']
    
    encoders = {}
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        # Define known categories for each feature
        if column == 'workclass':
            known_categories = ['Private', 'Self-emp', 'Gov', 'Others']
        elif column == 'marital-status':
            known_categories = ['Married', 'Single', 'Divorced', 'Separated']
        elif column == 'occupation':
            known_categories = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                              'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                              'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                              'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                              'Armed-Forces']
        elif column == 'relationship':
            known_categories = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried']
        elif column == 'race':
            known_categories = ['White', 'Black', 'Asian', 'Other']
        elif column == 'gender':
            known_categories = ['Male', 'Female']
        elif column == 'native-country':
            known_categories = ['United-States', 'Other']
            
        encoders[column].fit(known_categories)
        df[column] = encoders[column].transform(df[column])
    
    return df

# Build input DataFrame with all required features
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [200000],  # Default value
    'educational-num': [16 if education == "PhD" else 
                       14 if education == "Masters" else 
                       13 if education == "Bachelors" else 
                       12 if education == "HS-grad" else 11],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': ["Not-in-family"],  # Default value
    'race': [race],
    'gender': [gender],
    'capital-gain': [0],  # Default value
    'capital-loss': [0],  # Default value
    'hours-per-week': [hours_per_week],
    'native-country': ["United-States"]  # Default value
})

# Ensure column order matches model's expected features
input_df = input_df[expected_features]

st.write("### 🔎 Input Data")
st.write(input_df)
st.info("Note: Some features use default values for simplicity. For more accurate predictions, collect all feature values.")

def get_salary_range(prediction):
    if prediction == "<=50K":
        return "30,000 - 50,000 USD"
    else:
        return "50,000 - 100,000+ USD"

# Predict button
if st.button("Predict Salary Class"):
    try:
        # Create a copy to avoid modifying the display DataFrame
        prediction_df = input_df.copy()
        # Encode categorical features
        prediction_df = encode_categorical_features(prediction_df)
        
        prediction = model.predict(prediction_df)
        proba = model.predict_proba(prediction_df)
        
        st.success(f"✅ Prediction: {prediction[0]}")
        st.info(f"📊 Estimated Salary Range: {get_salary_range(prediction[0])}")
        st.write("Confidence Scores:")
        st.progress(float(proba[0][1]))
        st.write(f"Confidence: {proba[0][1]:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please check if all input values are valid.")

# Feature importance analysis
if st.button("Show Feature Analysis"):
    try:
        fig = go.Figure(go.Bar(
            x=feature_importance['Feature'],
            y=feature_importance['Importance'],
            text=np.round(feature_importance['Importance'], 3),
            textposition='auto',
        ))
        fig.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Features",
            yaxis_title="Importance Score"
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Unable to display feature importance: {str(e)}")
        st.info("This may occur if the model doesn't support feature importance analysis")

# Career recommendations function
def get_career_recommendations(education, occupation, experience):
    recommendations = {
        "Skills": [],
        "Next Role": "",
        "Education": ""
    }
    
    if experience < 5:
        recommendations["Skills"] = ["Leadership", "Project Management", "Technical Skills"]
        recommendations["Next Role"] = "Senior " + occupation
        recommendations["Education"] = "Consider Advanced Certification"
    else:
        recommendations["Skills"] = ["Strategic Planning", "Team Management", "Industry Expertise"]
        recommendations["Next Role"] = "Lead " + occupation
        recommendations["Education"] = "Consider MBA or Domain Specialization"
    
    return recommendations

# Add to UI after predictions
st.markdown("### 🎯 Career Growth Recommendations")
recommendations = get_career_recommendations(education, occupation, experience)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Recommended Skills**")
    for skill in recommendations["Skills"]:
        st.write(f"- {skill}")
with col2:
    st.write("**Next Career Move**")
    st.write(recommendations["Next Role"])
with col3:
    st.write("**Educational Path**")
    st.write(recommendations["Education"])

# Industry comparison function
def get_industry_comparison(occupation, hours_per_week):
    industry_averages = {
        "Tech-support": 45,
        "Sales": 40,
        "Exec-managerial": 50,
        "Prof-specialty": 42,
        "Others": 40
    }
    
    avg_hours = industry_averages.get(occupation, industry_averages["Others"])
    difference = hours_per_week - avg_hours
    
    return avg_hours, difference

# Add to UI
st.markdown("### 📊 Industry Comparison")
avg_hours, difference = get_industry_comparison(occupation, hours_per_week)

col1, col2 = st.columns(2)
with col1:
    st.metric("Industry Average Hours", f"{avg_hours}h/week", 
              f"{difference:+.1f}h difference")
with col2:
    if difference > 5:
        st.warning("⚠️ You're working more than industry average")
    elif difference < -5:
        st.info("💡 You're working less than industry average")
    else:
        st.success("✅ Your working hours are in line with industry average")

# Career timeline function

# Add file uploader widget
uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # Verify required columns
        missing_cols = set(expected_features) - set(batch_data.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Please ensure your CSV contains all required features.")
        else:
            # Ensure column order and encode categorical features
            batch_data = batch_data[expected_features]
            batch_data = encode_categorical_features(batch_data)
            
            st.write("Uploaded data preview:", batch_data.head())
            
            batch_preds = model.predict(batch_data)
            batch_data['PredictedClass'] = batch_preds
            st.write("✅ Predictions:")
            st.write(batch_data.head())
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, 
                             file_name='predicted_classes.csv', 
                             mime='text/csv')
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")

# Optionally, add a footer or additional info
st.markdown("---")
st.markdown("Made with Streamlit | © 2025")

def generate_career_roadmap(education, occupation, experience, salary_prediction):
    roadmap = {
        "Current": {
            "Role": occupation,
            "Level": "Entry" if experience < 3 else "Mid" if experience < 8 else "Senior",
            "Salary Range": get_salary_range(salary_prediction)
        },
        "1 Year": {
            "Skills": ["Technical Certifications", "Project Management", "Domain Expertise"],
            "Expected Role": f"Senior {occupation}",
            "Development": "Industry Certifications"
        },
        "3 Years": {
            "Skills": ["Leadership", "Team Management", "Strategic Planning"],
            "Expected Role": f"Lead {occupation}",
            "Development": "Management Training"
        },
        "5 Years": {
            "Skills": ["Executive Leadership", "Business Strategy", "Innovation"],
            "Expected Role": "Director/Manager Level",
            "Development": "MBA/Executive Education"
        }
    }
    
    # Create timeline visualization
    timeline_data = []
    current_year = 2025
    
    for stage, details in roadmap.items():
        if stage == "Current":
            continue
        timeline_data.append({
            "Stage": stage,
            "Role": details["Expected Role"],
            "Start": current_year,
            "End": current_year + (1 if stage == "1 Year" else 3 if stage == "3 Years" else 5)
        })
    
    return roadmap, timeline_data

# Add this after the prediction section
if st.button("Generate Career Roadmap"):
    try:
        prediction_df = input_df.copy()
        prediction_df = encode_categorical_features(prediction_df)
        prediction = model.predict(prediction_df)[0]
        
        roadmap, timeline = generate_career_roadmap(education, occupation, experience, prediction)
        
        st.success("🎯 Your Personalized Career Roadmap")
        
        # Current Status
        st.subheader("Current Position")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Role: {roadmap['Current']['Role']}")
        with col2:
            st.info(f"Level: {roadmap['Current']['Level']}")
        with col3:
            st.info(f"Salary Range: {roadmap['Current']['Salary Range']}")
        
        # Future Projections
        st.subheader("Career Progression Plan")
        tabs = st.tabs(["1 Year", "3 Years", "5 Years"])
        
        for i, (tab, stage) in enumerate(zip(tabs, ["1 Year", "3 Years", "5 Years"])):
            with tab:
                st.write(f"**Expected Role:** {roadmap[stage]['Expected Role']}")
                st.write("**Required Skills:**")
                for skill in roadmap[stage]['Skills']:
                    st.write(f"- {skill}")
                st.write(f"**Recommended Development:** {roadmap[stage]['Development']}")
                
    except Exception as e:
        st.error(f"Error generating roadmap: {str(e)}")

