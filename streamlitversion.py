import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define scopes
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if st.session_state.page == "Home":
    st.title("Welcome to the Data Analysis App")
    st.write("This application provides an in-depth analysis of the Google Sheets data.")
    st.write("Choose a section to navigate to for different functionalities:")
    st.write("- **Dashboard:** View data analysis visualizations and summaries.")
    st.write("- **Machine Learning:** Access machine learning models and predictions (Coming Soon).")
    st.write("- **Backend:** Backend management and settings (Coming Soon).")

    # Display navigation buttons after the explanation
    st.write("### Navigate to:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
    with col3:
        if st.button("Backend"):
            st.session_state.page = "Backend"

# Dashboard Page - Original Content
elif st.session_state.page == "Dashboard":
    st.title("Google Sheets Data Analysis")

    # Display 'Back to Home' button
    if st.button("Back to Home"):
        st.session_state.page = "Home"

    # Display data in Streamlit
    st.subheader("Data Preview")
    st.write(df.head())

    # 1. Pie Chart - Percentage of Employees by Department
    st.subheader("Percentage of Employees by Department")
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    plt.figure(figsize=(10, 6))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Employees by Department')
    plt.axis('equal')
    st.pyplot(plt)

    # (Include all remaining visualizations here...)

# Machine Learning Page - Placeholder Content
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
    st.write("Machine Learning content will be added here.")

# Backend Page - Placeholder Content
elif st.session_state.page == "Backend":
    st.title("Backend")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
    st.write("Backend management content will be added here.")
