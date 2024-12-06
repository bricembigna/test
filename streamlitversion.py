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
    st.write("- **Dashboard:** Explore an interactive HR dashboard offering comprehensive insights into your workforce, helping you make data-driven decisions.")
    st.write("- **Machine Learning:** Harness the power of AI to predict employee's income.")
    st.write("- **Data Management:** Efficiently manage employee records by adding, editing, or deleting information with ease.")


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


     # Income Statistics
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

##############

    
    st.subheader("Multiple Subplots: Age, Income, and Distance from Home")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    sns.histplot(df['Age'], bins=20, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title('Age Distribution')
    sns.histplot(df['MonthlyIncome'], bins=20, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Monthly Income Distribution')
    sns.histplot(df['DistanceFromHome'], bins=20, kde=True, ax=axs[2], color='orange')
    axs[2].set_title('Distance from Home Distribution')
    st.pyplot(fig)

    
    st.subheader("Department, Gender, and Job Role Insights")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))  # Create 1 row and 3 columns of subplots
    
    # 1. Pie Chart - Percentage of Employees by Department
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    axs[0].pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Employees by Department', fontsize=14)
    axs[0].axis('equal')  # Equal aspect ratio ensures a perfect circle
    
    # 2. Violin Plot - Age Distribution by Gender
    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True, ax=axs[1])
    axs[1].set_title('Age Distribution by Gender', fontsize=14)
    axs[1].set_xlabel('Gender')
    axs[1].set_ylabel('Age')
    
    # 3. Box Plot - Monthly Income by Job Role
    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df, ax=axs[2])
    axs[2].set_title('Monthly Income by Job Role', fontsize=14)
    axs[2].set_xlabel('Monthly Income')
    axs[2].set_ylabel('Job Role')
    
    # Common Title for the Subplots
    fig.suptitle('Key HR Insights', fontsize=16)
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Render Plot
    st.pyplot(fig)

##############

    st.subheader("Insights on Income, Age, and Department")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))  # Create 1 row and 3 columns of subplots
    
    # 1. Kernel Density Estimate Plot
    sns.kdeplot(data=df['MonthlyIncome'], fill=True, color='skyblue', alpha=0.5, ax=axs[0])
    axs[0].set_title('Density Plot of Monthly Income', fontsize=14)
    axs[0].set_xlabel('Monthly Income')
    axs[0].set_ylabel('Density')
    
    # 2. Joint Plot: Age vs. Monthly Income (Converted to Subplot)
    sns.scatterplot(x=df['Age'], y=df['MonthlyIncome'], ax=axs[1], alpha=0.6, color='blue')
    sns.regplot(x='Age', y='MonthlyIncome', data=df, ax=axs[1], scatter=False, color='red')
    axs[1].set_title('Age vs. Monthly Income', fontsize=14)
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Monthly Income')
    
    # 3. Grouped Bar Chart: Department vs. Business Travel
    grouped_data = df.groupby(['Department', 'BusinessTravel']).size().unstack(fill_value=0)
    grouped_data.plot(kind='bar', ax=axs[2], stacked=False, color=['lightblue', 'orange', 'green'])
    axs[2].set_title('Department vs. Business Travel', fontsize=14)
    axs[2].set_xlabel('Department')
    axs[2].set_ylabel('Number of Employees')
    axs[2].tick_params(axis='x', rotation=45)
    
    # Common Title for the Subplots
    fig.suptitle('Key Insights: Income, Age, and Department', fontsize=16)
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Render the Plot in Streamlit
    st.pyplot(fig)

 #################


    st.subheader("Department and Business Travel Insights")
    
    # Create a single figure with two subplots (side by side)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    # 1. Distribution of Employees by Department
    department_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axs[0])
    axs[0].set_title('Distribution of Employees by Department', fontsize=14)
    axs[0].set_xlabel('Department')
    axs[0].set_ylabel('Number of Employees')
    axs[0].tick_params(axis='x', rotation=45)
    
    # 2. Business Travel Frequency
    business_travel_counts = df['BusinessTravel'].value_counts()
    business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black', ax=axs[1])
    axs[1].set_title('Business Travel Frequency', fontsize=14)
    axs[1].set_xlabel('Business Travel Category')
    axs[1].set_ylabel('Number of Employees')
    axs[1].tick_params(axis='x', rotation=45)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Render the combined plot in Streamlit
    st.pyplot(fig)


    #################

   
    
    
    st.subheader("Joint Plot: Age vs. Monthly Income")
    fig = sns.jointplot(x='Age', y='MonthlyIncome', data=df, kind='reg', height=8, space=0.2)
    st.pyplot(fig)
    

    
    
   



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
        # Display 'Homepage' button
        if st.button("Homepage"):
            st.session_state.page = "Home"

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

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
            # Display 'Homepage' button
        if st.button("Homepage"):
            st.session_state.page = "Home"

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

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
            # Display 'Homepage' button
        if st.button("Homepage"):
            st.session_state.page = "Home"


    # Delete Employee Page
    def delete_employee_page():
        st.header("Delete Employee Data")
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number to Delete", employee_numbers)

        if selected_emp:
            emp_index = employee_numbers.index(selected_emp) + 1
            if st.button("Delete Employee"):
                worksheet.delete_rows(emp_index + 1)
                st.success(f"Employee {selected_emp} deleted successfully!")

        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        # Display 'Homepage' button
        if st.button("Homepage"):
            st.session_state.page = "Home"


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





######################################
import streamlit as st
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Define scopes
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Authenticate using Streamlit secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],  # Use your Streamlit secrets for service account JSON
    scopes=SCOPES,
    subject="user@yourdomain.com"  # Replace with an actual email in your Workspace domain
)
service = build("calendar", "v3", credentials=credentials)

# Function to create a calendar event
def create_event(summary, description, start_time, end_time):
    try:
        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "UTC",
            },
        }
        event_result = service.events().insert(calendarId="primary", body=event).execute()
        return event_result
    except Exception as e:
        st.error(f"Error creating event: {e}")
        return None

# Streamlit App UI
st.title("Google Calendar Event Scheduler")

st.subheader("Create a New Event")
# Event details input
event_summary = st.text_input("Event Title", "New Event")
event_description = st.text_area("Event Description", "Enter event description here...")
start_time = st.text_input("Start Time (YYYY-MM-DDTHH:MM:SS)", "2024-12-07T10:00:00")
end_time = st.text_input("End Time (YYYY-MM-DDTHH:MM:SS)", "2024-12-07T11:00:00")

# Schedule Event
if st.button("Schedule Event"):
    if event_summary and start_time and end_time:
        event = create_event(event_summary, event_description, start_time, end_time)
        if event:
            st.success(f"Event created successfully! [View Event]({event.get('htmlLink')})")
    else:
        st.warning("Please fill out all the required fields.")

