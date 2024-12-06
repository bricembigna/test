import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Authenticate with Google API using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=SCOPES
)
client = gspread.authorize(credentials)

# Access Google Sheet
SHEET_NAME = "Dataset"
sheet = client.open(SHEET_NAME).sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
def home_page():
    st.title("Welcome to the HR Monitor!")
    st.write(
        "This application provides you with in-depth analysis of your HR data in real time."
    )
    st.write(
        "Choose a section to navigate to for different functionalities:"
    )
    st.write("- **Dashboard:** Explore an interactive HR dashboard offering insights.")
    st.write("- **Machine Learning:** Predict employees' income using AI.")
    st.write("- **Data Management:** Manage employee records efficiently.")

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

# Dashboard Page
def dashboard_page():
    st.title("Google Sheets Data Analysis")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # Income Statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

    # Visualizations
    st.subheader("Age, Income, and Distance from Home Distributions")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(df['Age'], bins=20, kde=True, ax=axs[0])
    axs[0].set_title('Age Distribution')
    sns.histplot(df['MonthlyIncome'], bins=20, kde=True, ax=axs[1])
    axs[1].set_title('Monthly Income Distribution')
    sns.histplot(df['DistanceFromHome'], bins=20, kde=True, ax=axs[2])
    axs[2].set_title('Distance from Home Distribution')
    st.pyplot(fig)

    # Department Insights
    st.subheader("Department, Gender, and Job Role Insights")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))

    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    axs[0].pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Employees by Department')

    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True, ax=axs[1])
    axs[1].set_title('Age Distribution by Gender')

    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df, ax=axs[2])
    axs[2].set_title('Monthly Income by Job Role')

    fig.suptitle('Key HR Insights', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    # Additional Insights
    st.subheader("Department and Business Travel Insights")
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    department_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[0])
    axs[0].set_title('Distribution of Employees by Department')
    axs[0].set_xlabel('Department')
    axs[0].set_ylabel('Number of Employees')
    axs[0].tick_params(axis='x', rotation=45)

    business_travel_counts = df['BusinessTravel'].value_counts()
    business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black', ax=axs[1])
    axs[1].set_title('Business Travel Frequency')
    axs[1].set_xlabel('Business Travel Category')
    axs[1].set_ylabel('Number of Employees')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Joint Plot: Age vs. Monthly Income")
    joint_fig = sns.jointplot(x='Age', y='MonthlyIncome', data=df, kind='reg', height=8, space=0.2)
    st.pyplot(joint_fig)

# Machine Learning Page
def machine_learning_page():
    st.title("Machine Learning")
    if st.button("Homepage"):
        st.session_state.page = "Home"

    st.subheader("Predict Monthly Income")

    # Data Preparation
    df_ml = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
    X = df_ml[['TotalWorkingYears', 'JobLevel']]
    y = df_ml['MonthlyIncome']

    # Train-Test Split
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict Monthly Income
    user_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, step=1, value=10)
    user_job_level = st.number_input("Job Level", min_value=1, max_value=5, step=1, value=2)
    user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
    predicted_income = model.predict(user_input)[0]
    st.write(f"Predicted Monthly Income: ${predicted_income:.2f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_test['TotalWorkingYears'], model.predict(X_test), c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Predicted Monthly Income")
    ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Job Level")
    ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
    ax.legend()
    st.pyplot(fig)

# Backend Page
def backend_page():
    st.title("Data Input Manager")
    worksheet = client.open(SHEET_NAME).sheet1
    headers = worksheet.row_values(1)
    sheet_data = worksheet.get_all_values()

    st.write("Manage employee records:")

    def add_employee_page():
        st.header("Add New Employee Data")
        st.write("Enter details for the new employee:")

        employee_numbers = [
            int(row[headers.index("EmployeeNumber")])
            for row in sheet_data[1:]
            if row[headers.index("EmployeeNumber")].isdigit()
        ]
        next_employee_number = max(employee_numbers) + 1

        with st.form("add_employee_form"):
            age = st.selectbox("Age", list(range(18, 63)))
            attrition = st.selectbox("Attrition", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
            education = st.slider("Education Level", min_value=1, max_value=5)
            education_field = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])
            environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            job_role = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
            job_level = st.slider("Job Level", min_value=1, max_value=5)
            years_at_company = st.number_input("Years at Company", min_value=0, step=1)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, step=1)

            if st.form_submit_button("Submit"):
                new_employee = [
                    age, attrition, business_travel, daily_rate, department, distance_from_home,
                    education, education_field, environment_satisfaction, gender, job_role, job_level,
                    years_at_company, monthly_income, next_employee_number
                ]
                worksheet.append_row(new_employee)
                st.success(f"Employee added successfully with Employee Number {next_employee_number}!")

    def edit_employee_page():
        st.header("Edit Employee Data")

    def delete_employee_page():
        st.header("Delete Employee Data")

    st.write("Choose an action:")
    if st.button("Add New Employee Data"):
        add_employee_page()
    if st.button("Edit Employee Data"):
        edit_employee_page()
    if st.button("Delete Employee Data"):
        delete_employee_page()

# Page Navigation
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Dashboard":
    dashboard_page()
elif st.session_state.page == "Machine Learning":
    machine_learning_page()
elif st.session_state.page == "Backend":
    backend_page()
