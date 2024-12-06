import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define OAuth scopes for Google APIs
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"], scopes=SCOPES
)
client = gspread.authorize(credentials)

# Access Google Sheet
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------------------
# Home Page
# ---------------------------
if st.session_state.page == "Home":
    st.title("Welcome to the HR Monitor!")
    st.write("Analyze and manage HR data in real time.")
    
    st.write("### Features")
    st.write("- **Dashboard**: Interactive insights into your workforce data.")
    st.write("- **Machine Learning**: Predict employee income with AI.")
    st.write("- **Data Management**: Add, edit, or delete employee records.")

    # Navigation buttons
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

# ---------------------------
# Dashboard Page
# ---------------------------
elif st.session_state.page == "Dashboard":
    st.title("Dashboard: HR Data Insights")
    if st.button("Homepage"):
        st.session_state.page = "Home"

    # Income Statistics
    st.subheader("Income Statistics")
    mean_income = df["MonthlyIncome"].mean()
    median_income = df["MonthlyIncome"].median()
    std_income = df["MonthlyIncome"].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

    # Distribution Plots
    st.subheader("Distributions: Age, Income, and Distance from Home")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(df["Age"], bins=20, kde=True, ax=axs[0], color="skyblue").set_title("Age Distribution")
    sns.histplot(df["MonthlyIncome"], bins=20, kde=True, ax=axs[1], color="green").set_title("Monthly Income")
    sns.histplot(df["DistanceFromHome"], bins=20, kde=True, ax=axs[2], color="orange").set_title("Distance from Home")
    st.pyplot(fig)

    # Pie, Violin, and Box Plots
    st.subheader("Department, Gender, and Job Role Insights")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))
    department_counts = df["Department"].value_counts()
    axs[0].pie(department_counts, labels=department_counts.index, autopct="%1.1f%%", startangle=140)
    axs[0].set_title("Employees by Department")
    sns.violinplot(x="Gender", y="Age", data=df, hue="Gender", split=True, ax=axs[1]).set_title("Age by Gender")
    sns.boxplot(x="MonthlyIncome", y="JobRole", data=df, ax=axs[2]).set_title("Income by Job Role")
    plt.tight_layout()
    st.pyplot(fig)

    # Age vs. Monthly Income
    st.subheader("Age vs. Monthly Income")
    fig = sns.jointplot(x="Age", y="MonthlyIncome", data=df, kind="reg", height=8, space=0.2)
    st.pyplot(fig)

# ---------------------------
# Machine Learning Page
# ---------------------------
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning: Predict Monthly Income")
    if st.button("Homepage"):
        st.session_state.page = "Home"

    # Prepare data for model
    df_ml = df[["TotalWorkingYears", "JobLevel", "MonthlyIncome"]].dropna()
    X = df_ml[["TotalWorkingYears", "JobLevel"]]
    y = df_ml["MonthlyIncome"]

    # Train/test split and model training
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Input form for prediction
    st.subheader("Enter Details to Predict Income")
    years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
    job_level = st.number_input("Job Level (1-5)", min_value=1, max_value=5, value=2)

    user_data = pd.DataFrame({"TotalWorkingYears": [years], "JobLevel": [job_level]})
    predicted_income = model.predict(user_data)[0]
    st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")

# ---------------------------
# Backend Page
# ---------------------------
elif st.session_state.page == "Backend":
    st.title("Data Management: Employee Records")
    worksheet = client.open("Dataset").sheet1
    headers = worksheet.row_values(1)

    # Subpage navigation
    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    def main_page():
        st.subheader("Main Actions")
        if st.button("Add Employee"):
            st.session_state.Backend_subpage = "add"
        if st.button("Edit Employee"):
            st.session_state.Backend_subpage = "edit"
        if st.button("Delete Employee"):
            st.session_state.Backend_subpage = "delete"

    # Subpages for managing employee records
    def add_employee():
        st.subheader("Add New Employee")
        form_data = {header: st.text_input(header) for header in headers}
        if st.button("Submit"):
            worksheet.append_row(list(form_data.values()))
            st.success("Employee added successfully!")
            st.session_state.Backend_subpage = "main"

    def edit_employee():
        st.subheader("Edit Employee Data")
        emp_id = st.text_input("Enter Employee Number")
        if st.button("Load Data"):
            # Fetch and allow editing of employee data here
            pass

    def delete_employee():
        st.subheader("Delete Employee")
        emp_id = st.text_input("Enter Employee Number")
        if st.button("Delete"):
            # Logic to delete employee
            pass

    # Render the correct subpage
    if st.session_state.Backend_subpage == "main":
        main_page()
    elif st.session_state.Backend_subpage == "add":
        add_employee()
    elif st.session_state.Backend_subpage == "edit":
        edit_employee()
    elif st.session_state.Backend_subpage == "delete":
        delete_employee()
