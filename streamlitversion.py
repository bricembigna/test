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
    st.title("Welcome to the HR monitor!")
    st.write("This application provides you with in-depth analysis of your HR data in real time.")
    st.write("Choose a section to navigate to for different functionalities:")
    st.write("- **Dashboard:** Check your HR dashboards.")
    st.write("- **Machine Learning:** Leverage AI to make predictions about your workforce (Coming Soon).")
    st.write("- **Backend:** Backend management and settings (Coming Soon).")

    # Display navigation buttons under the explanation
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

# Dashboard Page - Complete Data Analysis Content
elif st.session_state.page == "Dashboard":
    st.title("Google Sheets Data Analysis")

    # Display 'Homepage' button
    if st.button("Homepage"):
        st.session_state.page = "Home"


     # 5. Income Statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")


    
    # 1. Pie Chart - Percentage of Employees by Department
    st.subheader("Percentage of Employees by Department")
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    plt.figure(figsize=(10, 6))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Employees by Department')
    plt.axis('equal')
    st.pyplot(plt)

    # 2. Violin Plot - Age Distribution by Gender
    st.subheader("Age Distribution by Gender")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True)
    plt.title('Age Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Age')
    st.pyplot(plt)

    # 3. Histogram - Age Distribution
    st.subheader("Age Distribution Histogram (Ages 18 to 60)")
    bins = list(range(18, 61))
    plt.figure(figsize=(12, 6))
    plt.hist(df['Age'], bins=bins, edgecolor='white', color='skyblue', alpha=0.7, align='left')
    plt.title('Age Distribution Histogram (Individual Ages 18 to 60)')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.xticks(bins)
    st.pyplot(plt)

    # 4. Bar Chart - Distribution of Employees by Department
    st.subheader("Distribution of Employees by Department")
    plt.figure(figsize=(10, 6))
    department_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Employees by Department')
    plt.xlabel('Department')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45)
    st.pyplot(plt)


    # 6. Business Travel Frequency
    st.subheader("Business Travel Frequency")
    business_travel_counts = df['BusinessTravel'].value_counts()
    plt.figure(figsize=(8, 5))
    business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Business Travel Frequency')
    plt.xlabel('Business Travel Category')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # 7. Pie Chart - Business Travel Distribution
    st.subheader("Business Travel Distribution")
    plt.figure(figsize=(8, 8))
    plt.pie(business_travel_counts, labels=business_travel_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Business Travel Distribution')
    plt.axis('equal')
    st.pyplot(plt)

    # 8. Violin Plot - Monthly Income by Job Role
    st.subheader("Distribution of Monthly Income by Job Role")
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='JobRole', x='MonthlyIncome', data=df, scale='count', inner='stick')
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 9. Boxen Plot - Monthly Income by Job Role
    st.subheader("Monthly Income by Job Role (Boxen Plot)")
    plt.figure(figsize=(12, 8))
    sns.boxenplot(y='JobRole', x='MonthlyIncome', data=df)
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 10. Box Plot - Monthly Income by Job Role
    st.subheader("Monthly Income by Job Role (Box Plot)")
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df)
    plt.title('Distribution of Monthly Income by Job Role')
    plt.xlabel('Monthly Income')
    plt.ylabel('Job Role')
    st.pyplot(plt)

    # 11. Violin Plot - Work-Life Balance by Job Role
    st.subheader("Distribution of Work-Life Balance by Job Role")
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='JobRole', x='WorkLifeBalance', data=df, inner='quartile')
    plt.title('Distribution of Work-Life Balance by Job Role')
    plt.xlabel('Work-Life Balance')
    plt.ylabel('Job Role')
    st.pyplot(plt)

# Machine Learning Page - Placeholder Content
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    if st.button("Homepage"):
        st.session_state.page = "Home"


    # Import necessary libraries for data manipulation, visualization, and modeling
    import pandas as pd  # For working with tabular data
    import numpy as np  # For numerical operations (not used here directly but commonly useful)
    import streamlit as st  # For creating an interactive web app
    from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
    from sklearn.linear_model import LinearRegression  # For performing linear regression
    from sklearn.metrics import mean_squared_error, r2_score  # For evaluating the regression model
    import matplotlib.pyplot as plt  # For creating plots
    import seaborn as sns  # For enhanced data visualization (not used directly but useful for customization)
    import gspread
    from google.oauth2.service_account import Credentials
    
    # Define scopes
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    
    # Authenticate using Streamlit secrets
    credentials = Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=scopes
    )
    client = gspread.authorize(credentials)
    
    # Access Google Sheet
    try:
        sheet = client.open("Dataset").sheet1  # Replace "Dataset" with your Google Sheet name
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error accessing Google Sheet: {e}")
        st.stop()

    #Sorting only the necessary data and error if not found
    try:
        # Selektiere nur die relevanten Spalten für die Regression
        df = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
    except KeyError as e:
        st.error(f"Fehlende Spalten: {e}")
        st.stop()

    # X and y data for and training them 
    X = df[['TotalWorkingYears', 'JobLevel']]
    y = df['MonthlyIncome']
    
    # Datenaufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Regressiontraining through sklearn formula
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction with method
    y_pred = model.predict(X_test)
    
    
    # Userinterface: using input of working years (max 40) and Job Level(1-5) with streamlit
    st.subheader("Predict Monthly Income")
    user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
    user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)
    
    # Prediction with user input data while making data frame with entered variables 
    user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
    predicted_income = model.predict(user_input)[0]
    st.write(f"Predicted Monthly Income: *{predicted_income:.2f}*")
    
    # Visualization: Scatterplot with farbcoded joblevel for better overview, the x-axis displays total working years and the y-axis shows predicted income, 
    # with point colors assigned via the viridis colormap for clear differentiation of job levels + color bar is added to show the job levels
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Predicted Monthly Income")
    ax.set_title("Predicted Monthly Income vs. Total Working Years (Color: Job Level)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Job Level")
    
    # Point of user output on graph and showing graph
    ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
    ax.legend()
    st.pyplot(fig)

# Backend Page - Data Input Manager
elif st.session_state.page == "Backend":
    st.title("Data Input Manager")

    # Access Google Sheets
    spreadsheet_name = "Dataset"  # Replace with your actual sheet name
    worksheet = client.open(spreadsheet_name).sheet1
    headers = worksheet.row_values(1)  # Get header row
    sheet_data = worksheet.get_all_values()  # Get all sheet data

    # Main Page
    def main_page():
        st.header("Main Page")
        st.write("Choose an action:")
        if st.button("Add New Employee Data"):
            st.session_state.Backend_subpage = "add_employee"
        if st.button("Change Employee Data"):
            st.session_state.Backend_subpage = "change_employee"
        if st.button("Delete Employee Data"):
            st.session_state.Backend_subpage = "delete_employee"

    # Add Employee Page
    def add_employee_page():
        st.header("Add New Employee Data")
        st.write("Enter details for the new employee:")

        # Automatically calculate the next Employee Number
        employee_numbers = [
            int(row[headers.index("EmployeeNumber")])
            for row in sheet_data[1:]
            if row[headers.index("EmployeeNumber")].isdigit()
        ]
        next_employee_number = max(employee_numbers) + 1

        # Input form
        with st.form("add_employee_form"):
            age = st.selectbox("Age", list(range(18, 63)))
            attrition = st.selectbox("Attrition", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
            education = st.slider("Educaiton level", min_value=1, max_value=5)
            education_field = st.selectbox("Education Field", [ "Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])
            employee_count = st.number_input("Employee Count", min_value=1, step=1)
            employee_number = next_employee_number
            environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            hourly_rate = st.number_input("Hourly Rate", min_value=0, step=1)
            job_involvement = st.slider("Job Involvement", min_value=1, max_value=4)
            job_level = st.slider("Job Level", min_value=1, max_value=5)
            job_role = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            monthly_income = st.number_input("Monthly Income ($)", min_value=1009, step=1)
            monthly_rate = st.number_input("Monthly Rate", min_value=1000, step=100)
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, step=1)
            over_18 = st.selectbox("Over 18", ["Y", "N"])
            overtime = st.selectbox("Overtime", ["Yes", "No"])
            percent_salary_hike = st.slider("Percent Salary Hike", min_value=0, max_value=100)
            performance_rating = st.slider("Performance Rating", min_value=1, max_value=5)
            relationship_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=4)
            standard_hours = st.number_input("Standard Hours", min_value=1, step=1)
            stock_option_level = st.number_input("Stock Option Level", min_value=0, step=1)
            total_working_years = st.number_input("Total Working Years", min_value=0, step=1)
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, step=1)
            work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=4)
            years_at_company = st.number_input("Years at Company", min_value=0, step=1)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, step=1)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, step=1)
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, step=1)

            if st.form_submit_button("Submit"):
                new_employee = [
                    age, attrition, business_travel, daily_rate, department, distance_from_home,
                    education, education_field, employee_count, employee_number, environment_satisfaction,
                    gender, hourly_rate, job_involvement, job_level, job_role, job_satisfaction, marital_status,
                    monthly_income, monthly_rate, num_companies_worked, over_18, overtime, percent_salary_hike,
                    performance_rating, relationship_satisfaction, standard_hours, stock_option_level,
                    total_working_years, training_times_last_year, work_life_balance, years_at_company,
                    years_in_current_role, years_since_last_promotion, years_with_curr_manager
                ]
                worksheet.append_row(new_employee)
                st.success(f"Employee added successfully with Employee Number {employee_number}!")

        if st.button("Previous page"):
            st.session_state.Backend_subpage = "main"

    # Change Employee Page
    def change_employee_page():
        st.header("Change Employee Data")
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number", employee_numbers)

        if selected_emp:
            emp_index = employee_numbers.index(selected_emp) + 1
            current_data = sheet_data[emp_index]

            with st.form("change_employee_form"):
                updated_employee = [
                    st.text_input(headers[i], value=current_data[i])
                    for i in range(len(headers))
                ]
                if st.form_submit_button("Update"):
                    worksheet.update(f"A{emp_index + 1}", [updated_employee])
                    st.success("Employee data updated successfully!")

        if st.button("Previous page"):
            st.session_state.Backend_subpage = "main"


    # Delete Employee Page
    def delete_employee_page():
        st.header("Delete Employee Data")
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number to Delete", employee_numbers)

        if selected_emp:
            emp_index = employee_numbers.index(selected_emp) + 1
            if st.button("Delete"):
                worksheet.delete_rows(emp_index + 1)
                st.success(f"Employee {selected_emp} deleted successfully!")

        if st.button("Previous page"):
            st.session_state.Backend_subpage = "main"


    # Render Subpages
    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    if st.session_state.Backend_subpage == "main":
        main_page()
    elif st.session_state.Backend_subpage == "add_employee":
        add_employee_page()
    elif st.session_state.Backend_subpage == "change_employee":
        change_employee_page()
    elif st.session_state.Backend_subpage == "delete_employee":
        delete_employee_page()

    # Display 'Homepage' button
     if st.button("Homepage"):
          st.session_state.page = "Home"
