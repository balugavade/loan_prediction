import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load your dataset
# data = pd.read_csv('your_dataset.csv')

# Dummy data
data = pd.DataFrame({
    'ApplicantIncome': [5000, 6000, 7000],
    'CoapplicantIncome': [0, 1500, 2000],
    'LoanAmount': [100, 200, 300],
    'Loan_Amount_Term': [360, 120, 240],
    'Credit_History': [1, 0, 1],
    'Loan_Status': [1, 0, 1]
})

# Define input and output variables
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Train a K-Nearest Neighbors model
kn_clf = KNeighborsClassifier(n_neighbors=1)
kn_clf.fit(X_train, y_train)

# Streamlit GUI
st.title("Loan Prediction")

st.sidebar.header("Input Features")

def get_user_input():
    ApplicantIncome = st.sidebar.number_input("Applicant Income", 0, 100000)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", 0, 50000)
    LoanAmount = st.sidebar.number_input("Loan Amount", 0, 100000)
    Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", 0, 360)
    Credit_History = st.sidebar.selectbox("Credit History", (0, 1))

    user_data = {
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

# Display user inputs
st.write("User Input Features:")
st.write(user_input)

# Prediction
rf_prediction = rf_clf.predict(user_input)
kn_prediction = kn_clf.predict(user_input)

# Display predictions
st.subheader("Prediction by Random Forest")
st.write("Loan Approved" if rf_prediction[0] == 1 else "Loan Rejected")

st.subheader("Prediction by K-Nearest Neighbors")
st.write("Loan Approved" if kn_prediction[0] == 1 else "Loan Rejected")
