import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="Credit Loan Predictor", layout="centered")

st.title("Welcome to the Loan Eligibility Explorer")
st.write("""
This interactive app uses a Random Forest model to help you explore loan eligibility predictions. 
You can adjust model parameters to see how they affect the outcome.
""")
st.write("""
Built for learners, data enthusiasts, and aspiring data scientists.
""")

st.caption("📌 Expand the '>>' icon at the top-left to configure model settings.")

# Load dataset and preprocess
data_path = "data/raw/credit.csv"
df = load_and_preprocess_data(data_path)
X, y = create_dummy_vars(df)

# Sidebar: Model Configuration
st.sidebar.header("Model Parameter Section")
n_estimators = st.sidebar.number_input(
    "Number of Trees", min_value=1, max_value=100, value=5,
    help="Controls the number of trees in the forest. More trees = better performance (usually).")
max_depth = st.sidebar.number_input(
    "Maximum Depth", min_value=1, max_value=20, value=2,
    help="Limits how deep each tree can grow to avoid overfitting.")
max_features = st.sidebar.number_input(
    "Max Features", min_value=1, max_value=X.shape[1], value=7,
    help="Number of features considered at each split. Helps control overfitting")

# Train model using configured parameters
model, _, _ = train_RFmodel(X, y, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

# Evaluate model accuracy on a holdout test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
accuracy, _ = evaluate_model(model, X_test, y_test)


# Main Form for User Input
with st.form("user_inputs"):
    st.subheader("📝 Loan Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Marital Status", ["Married", "Not Married"])
        Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0, step=1000)
        CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0, step=1000)
        LoanAmount = st.number_input("Loan Amount", min_value=1000, step=1000)
        Loan_Amount_Term = st.selectbox("Loan Term (Months)", ["360", "180", "240", "120", "60"])
        Credit_History = st.selectbox("Credit History", ["1", "0"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Eligibility")

# Handle dummy variable conversion and prediction
if submitted:
    Gender_Male = 0 if Gender == "Female" else 1
    Gender_Female = 1 if Gender == "Female" else 0
    Married_Yes = 1 if Married == "Married" else 0
    Married_No = 1 if Married == "Not Married" else 0
    Dependents_0 = 1 if Dependents == "0" else 0
    Dependents_1 = 1 if Dependents == "1" else 0
    Dependents_2 = 1 if Dependents == "2" else 0
    Dependents_3 = 1 if Dependents == "3+" else 0
    Education_Graduate = 1 if Education == "Graduate" else 0
    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
    Self_Employed_No = 1 if Self_Employed == "No" else 0
    Property_Area_Rural = 1 if Property_Area == "Rural" else 0
    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0
    Loan_Amount_Term = int(Loan_Amount_Term)
    Credit_History = int(Credit_History)

    prediction_input = [[ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
        Married_No, Married_Yes, Dependents_0, Dependents_1,
        Dependents_2, Dependents_3, Education_Graduate,
        Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
        Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
    ]]

    new_prediction = model.predict(prediction_input)

    st.subheader("Prediction Result")
    if new_prediction[0] == 'Y':
        st.success("✅ You are eligible for the loan!")
    else:
        st.error("❌ Sorry, you are not eligible for the loan.")
    st.caption(f"Model Accuracy: **{accuracy * 100:.2f}%**")

# Optional: Feature Importance image
st.markdown("---")
st.subheader("What Influences Eligibility and How Our Model Picks It Up")
st.info(f"Your customized RF model used: **n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}**")
st.caption(f"Note: GridSearchCV was initially used offline to identify the best parameters. These have been set as the default values. You can now adjust them manually to test other configurations.")
st.caption(f"The best parameters were n_estimators = 5, max_depth = 2, and max_features = 7 with model accuracy of **78.86%**")
st.image("feature_importance.png", use_container_width=True)
st.caption(f"Note: This app is a classroom project designed to show how machine learning models work. You can test different parameter settings and observe how the model’s behavior changes. It’s a great way to understand real-world deployment concepts.")

# Footer
st.markdown("---")
st.caption("Built by Mary Grace Lunar | Educational Use Only")