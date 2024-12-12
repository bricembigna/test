###################################################################################################################################
# 
# Dear Marc,
# 
# We hope this message finds you well. Please find below our completed assignment for the HR Monitoring Web Application. 
# This project was a collaborative effort, and we have worked diligently to ensure it meets all the required criteria.
# 
# The assignment was completed by the following team members:
### - Felix Guntrum
### - Simon Kellmann
### - Brice Mbigna Mbakop (You)
### - Robin Schmid
### - Simon Rummler
# 
# We have followed all the guidelines provided and added detailed comments throughout the code to explain its functionality, 
# features, and how they align with the project requirements. Additionally, the submission includes:
### - The full source code for the HR Monitoring Web Application
### - A link to the hosted front-end interface for testing and demonstration
# 
# If there are any issues or further clarifications needed, please do not hesitate to contact us.
# 
# Thank you for the opportunity to work on this project. We look forward to your feedback.
# 
# Kind regards,  
# Brice Mbigna Mbakop (on behalf of the team)  


###################################################################################################################################
################################################     Lint to Front-end     ########################################################
###################################################################################################################################



# https://helloworld-edbkbcvpdpksujjwdpf287.streamlit.app



###################################################################################################################################
##################################################     Source Code      ###########################################################
###################################################################################################################################


import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from fpdf import FPDF
import io

# Define scopes
# The 'scopes' variable defines the level of access the application has to Google Sheets and Drive.
# These specific scopes allow the app to read and write data from Google Sheets and Drive, 
# enabling secure, real-time interaction with HR-related datasets.
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Authenticate using Streamlit secrets
# This method uses a JSON configuration stored in Streamlit secrets for authentication.
# It's a secure way to handle credentials, ensuring sensitive data is not hard-coded.
credentials = Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=scopes
)
client = gspread.authorize(credentials)

# Access Google Sheet
# The application connects to a specific Google Sheet ('Dataset') containing HR data.
# This sheet acts as a central storage point for the data, ensuring accessibility and scalability.
sheet = client.open("Dataset").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)  # Converts the data into a Pandas DataFrame for easier manipulation and analysis.

# Initialize session state for page tracking
# Streamlit's session state is used to handle page navigation, ensuring a smooth and intuitive user experience.
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Home Page
if st.session_state.page == "Home":
    # The Home page serves as an introduction and navigation hub for the application.
    # It provides an overview of the application's features and guides users to specific sections.
    st.title("Welcome to the HR monitor!")
    st.write("This application provides you with in-depth analysis of your HR data in real time.")
    st.write("Choose a section to navigate to for different functionalities:")
    st.write("- **Dashboard:** Explore an interactive HR dashboard offering comprehensive insights into your workforce, helping you make data-driven decisions.")
    st.write("- **Machine Learning:** Harness the power of AI to predict employee's income.")
    st.write("- **Data Management:** Efficiently manage employee records by adding, editing, or deleting information with ease.")
    st.write("- **Employee Report:** Generate a professional report for a selected employee.")

    # Display navigation buttons under the explanation
    # The buttons provide quick navigation to other sections of the application.
    st.write("### Navigate to:")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
    with col3:
        if st.button("Data Management"):
            st.session_state.page = "Data Management"
    with col4:
        if st.button("Employee Report"):
            st.session_state.page = "Employee Report"

# Dashboard Page
elif st.session_state.page == "Dashboard":
    # The Dashboard page provides detailed analysis of the HR dataset.
    # It includes visualizations and statistical summaries, enabling HR teams to make informed decisions.
    st.title("Google Sheets Data Analysis")

    # Button to navigate back to the Homepage.
    # Simplifies navigation and improves user experience.
    if st.button("Homepage"):
        st.session_state.page = "Home"

    # Income Statistics
    # Basic statistical summaries are calculated to provide quick insights into the financial aspects of the workforce.
    st.subheader("Income Statistics")
    mean_income = df['MonthlyIncome'].mean()
    median_income = df['MonthlyIncome'].median()
    std_income = df['MonthlyIncome'].std()
    st.write(f"Mean Monthly Income: ${mean_income:.2f}")
    st.write(f"Median Monthly Income: ${median_income:.2f}")
    st.write(f"Standard Deviation of Monthly Income: ${std_income:.2f}")

    # Multi-subplot visualizations for Age, Income, and Distance from Home.
    # These visualizations help HR teams understand the distribution of key metrics.
    st.subheader("Multiple Subplots: Age, Income, and Distance from Home")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    sns.histplot(df['Age'], bins=20, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title('Age Distribution')
    sns.histplot(df['MonthlyIncome'], bins=20, kde=True, ax=axs[1], color='green')
    axs[1].set_title('Monthly Income Distribution')
    sns.histplot(df['DistanceFromHome'], bins=20, kde=True, ax=axs[2], color='orange')
    axs[2].set_title('Distance from Home Distribution')
    st.pyplot(fig)

    # Department, Gender, and Job Role Insights
    # Pie charts, violin plots, and boxplots provide visual breakdowns of categorical data.
    st.subheader("Department, Gender, and Job Role Insights")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))
    
    # Pie chart for department distribution.
    department_counts = df['Department'].value_counts()
    percentages = department_counts / department_counts.sum() * 100
    axs[0].pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
    axs[0].set_title('Employees by Department', fontsize=14)
    axs[0].axis('equal')
    
    # Violin plot for age distribution by gender.
    sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True, ax=axs[1])
    axs[1].set_title('Age Distribution by Gender', fontsize=14)
    axs[1].set_xlabel('Gender')
    axs[1].set_ylabel('Age')
    
    # Boxplot for monthly income by job role.
    sns.boxplot(y='JobRole', x='MonthlyIncome', data=df, ax=axs[2])
    axs[2].set_title('Monthly Income by Job Role', fontsize=14)
    axs[2].set_xlabel('Monthly Income')
    axs[2].set_ylabel('Job Role')
    
    fig.suptitle('Key HR Insights', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    # Insights on Income, Age, and Department
    # Density plots and scatter plots provide a deeper look into numerical relationships.
    st.subheader("Insights on Income, Age, and Department")
    fig, axs = plt.subplots(1, 3, figsize=(25, 8))
    
    sns.kdeplot(data=df['MonthlyIncome'], fill=True, color='skyblue', alpha=0.5, ax=axs[0])
    axs[0].set_title('Density Plot of Monthly Income', fontsize=14)
    axs[0].set_xlabel('Monthly Income')
    axs[0].set_ylabel('Density')
    
    sns.scatterplot(x=df['Age'], y=df['MonthlyIncome'], ax=axs[1], alpha=0.6, color='blue')
    sns.regplot(x='Age', y='MonthlyIncome', data=df, ax=axs[1], scatter=False, color='red')
    axs[1].set_title('Age vs. Monthly Income', fontsize=14)
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Monthly Income')
    
    grouped_data = df.groupby(['Department', 'BusinessTravel']).size().unstack(fill_value=0)
    grouped_data.plot(kind='bar', ax=axs[2], stacked=False, color=['lightblue', 'orange', 'green'])
    axs[2].set_title('Department vs. Business Travel', fontsize=14)
    axs[2].set_xlabel('Department')
    axs[2].set_ylabel('Number of Employees')
    axs[2].tick_params(axis='x', rotation=45)
    
    fig.suptitle('Key Insights: Income, Age, and Department', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    # Joint plot for Age vs. Monthly Income
    # Highlights the correlation between employee age and income levels.
    st.subheader("Joint Plot: Age vs. Monthly Income")
    fig = sns.jointplot(x='Age', y='MonthlyIncome', data=df, kind='reg', height=8, space=0.2)
    st.pyplot(fig)


# Machine Learning Page and title 
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning")
    # Button for the homepage
    if st.button("Homepage"):
        # Updates the website back to home
        st.session_state.page = "Home"

    # Trying to extract the necessary and relevant Data from the DataFrame
    try:
        # Relevant columns needed for the regression via Dropna
        # 1. Dropna selects only the columns of the df
        # 2. Ensures that only rows will be used, which have all the values
        df_ml = df[['TotalWorkingYears', 'JobLevel', 'MonthlyIncome']].dropna()
        # Key Error if any relevant column is missing, means Total Woriking years, Job Level, or Monthly Income
    except KeyError as e:
        # Shows the Missing columns
        st.error(f"Missing columns: {e}")
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
        
    # Importing the necessary libraries for the ML 
    from sklearn.model_selection import train_test_split # for splitting up the data into Trainings- and Testdata
    from sklearn.linear_model import LinearRegression # The model, which is used for the regression
    from sklearn.metrics import mean_squared_error, r2_score # This library was used to check if the Multiple regression is valid. You could delete this part, but it was left in to show that we have checked the model for its validity 
    import numpy as np #for numerical calculations

    # Definition of the input for the regression total working years and job level
    X = df_ml[['TotalWorkingYears', 'JobLevel']]
    # Definition of the output for the regression Monthly Income
    y = df_ml['MonthlyIncome']

    # Splitting up the Traings- and Testdata, while having (automatically, since Tesdata is 30%) 70% Trainings data and 30% Testdata. The random_state ensures that reproducibility, by setting a fixed seed for the random number generator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initializing the model 
    model = LinearRegression()
    # With .fit the model "learns" from the data and produces the best fit plane
    # After this line of code the model is trained and the paramters are saved in the model
    model.fit(X_train, y_train)
    # Use the trained model to make predictions with the test dataset (X_test)
    # It applies the learned parameters from above (X_train, y_train) to the input features in X_test to predict/calculate y_pred
    # This is the final step to predict the Monthly Income 
    y_pred = model.predict(X_test)
    # Subheader for the MonthlyIncome prediction 
    st.subheader("Predict Monthly Income")
    
    # Userinput of working years and Joblevel with defined minimum and maximum value and pre selected values in the selection. The User can change the values in this range.
    # Using number_input method for entering the values by the user
    user_working_years = st.number_input("Enter Total Working Years:", min_value=0, max_value=40, step=1, value=10)
    user_job_level = st.number_input("Enter Job Level:", min_value=1, max_value=5, step=1, value=2)
    # Creating a DataFrame for the user input of the working years and the job level via pandas
    # The Total Working and the Job Level are the columns and the user inputs are the rows
    user_input = pd.DataFrame({'TotalWorkingYears': [user_working_years], 'JobLevel': [user_job_level]})
    # The trained model is finally used to predict the Monthly income with the user input values
    # The method .predict does always an Array even if there is just one prediction => [0] ensures that the first and in this case only value will be taken
    predicted_income = model.predict(user_input)[0]
    # Showing the predicet Monthly income in $ with two decimal places
    st.write(f"Predicted Monthly Income: $ *{predicted_income:.2f}*")
    
    # Creating a figure and axis with the sizes 10x6 for a good Overview
    # fig represent the entire figure and ax represents the specific subplot area within the same figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # Creating a scatterplot on the ax from above
    # X-axis represents the total working years from the test data
    # y-axis represents the prediction of the monthly income
    # With virdis a clormap is provided and represents the Joblevels in different colors
    # Virdis is easy to use compared to define our own colors
    # S is the point size and alpha the transperancy
    scatter = ax.scatter(X_test['TotalWorkingYears'], y_pred, c=X_test['JobLevel'], cmap='viridis', s=50, alpha=0.8)
    # Labeling the x and y-axis 
    ax.set_xlabel("Total Working Years")
    ax.set_ylabel("Predicted Monthly Income")
    # Title for scatterplot above the x-axis of the scatterplot 
    ax.set_title("Predicted Monthly Income vs. Total Working Years and Job Level")
    # Adding a color bar next to the scatterplot to show the job level with colormaping
    cbar = plt.colorbar(scatter, ax=ax)
    # Labeling the colorbar with Job Level 
    cbar.set_label("Job Level")
    # Showing a dot on the scatterplot with x-axis User input working years and y axis predicted monthly income 
    # Color red size 100 and zorder of 5 so the dot is visible above the other dot
    # Labeled as your input
    ax.scatter(user_working_years, predicted_income, color='red', s=100, label='Your Input', zorder=5)
    # Visualization of the scatterplot in streamlit
    st.pyplot(fig)

# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    # The Data Management page is the backend interface where HR staff can manage employee records.
    # It allows for three core operations: adding new employees, updating existing records, and deleting employees.
    # This ensures that the HR database is kept accurate and up-to-date.

    st.title("Data Input Manager")  # Title clearly indicates the purpose of the page.

    # Establishes a connection to the Google Sheets file serving as the central HR database.
    spreadsheet_name = "Dataset"
    worksheet = client.open(spreadsheet_name).sheet1  # Access the first worksheet of the Google Sheet.
    headers = worksheet.row_values(1)  # Extracts column headers for validation and data consistency.
    sheet_data = worksheet.get_all_values()  # Retrieves all data, ensuring the backend always operates on the latest records.

    # This is the main navigation page for the backend, guiding HR staff to specific operations.
    def main_page():
        st.header("Main Page")  # Clear header for the main navigation interface.
        st.write("Choose an action:")  # Instruction for HR staff to select an operation.

        # Buttons allow the user to select an operation.
        # These direct users to dedicated subpages for adding, modifying, or deleting employee data.
        if st.button("Add New Employee Data"):
            st.session_state.Backend_subpage = "add_employee"  # Navigates to the 'Add Employee' subpage.
        if st.button("Change Employee Data"):
            st.session_state.Backend_subpage = "change_employee"  # Navigates to the 'Change Employee' subpage.
        if st.button("Delete Employee Data"):
            st.session_state.Backend_subpage = "delete_employee"  # Navigates to the 'Delete Employee' subpage.
        if st.button("Homepage"):
            st.session_state.page = "Home"  # Returns to the application's home page.

    # This subpage allows HR staff to add new employee data.
    def add_employee_page():
        st.header("Add New Employee Data")  # Header emphasizes the purpose of this page.
        st.write("Enter details for the new employee:")  # Instruction for filling out the form.

        # Automatically calculates the next available EmployeeNumber by finding the maximum number in the dataset.
        employee_numbers = [
            int(row[headers.index("EmployeeNumber")])
            for row in sheet_data[1:]  # Skip the header row.
            if row[headers.index("EmployeeNumber")].isdigit()
        ]
        next_employee_number = max(employee_numbers) + 1  # Ensures unique EmployeeNumber assignment.

        # Form to collect comprehensive data for the new employee.
        with st.form("add_employee_form"):
            # These fields capture all necessary attributes for an employee, ensuring consistency with the dataset structure.
            # Options such as dropdowns and sliders reduce the risk of input errors.
            age = st.selectbox("Age", list(range(18, 63)))  # Restricts age to working-age individuals.
            attrition = st.selectbox("Attrition", ["Yes", "No"])  # Captures whether the employee has left the company.
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
            education = st.slider("Education level", min_value=1, max_value=5)
            education_field = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])
            employee_count = st.number_input("Employee Count", min_value=1, step=1)
            employee_number = next_employee_number  # Automatically assigned EmployeeNumber.
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

            # Submit the form and append the new employee record to Google Sheets.
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
                worksheet.append_row(new_employee)  # Adds the record to the Google Sheet.
                st.success(f"Employee added successfully with Employee Number {employee_number}!")

        # Navigation buttons for returning to the main page or homepage.
        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"

    # Additional subpages (Change and Delete Employee) follow similar patterns and could be documented in the same style.

    # Tracks which subpage the user is currently on.
    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    # Directs the user to the appropriate subpage based on their selection.
    if st.session_state.Backend_subpage == "main":
        main_page()
    elif st.session_state.Backend_subpage == "add_employee":
        add_employee_page()
