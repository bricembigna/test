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
# The application connects to a specific Google Sheet ('Dataset') containing HR data
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
    # The Home page introduces the football analytics application
    # and serves as the main navigation hub.

    st.title("⚽ Football Club Performance Monitor")

    st.write(
        "This application provides a comprehensive overview of player performance, "
        "training engagement, and team composition across all divisions of the club."
    )

    st.write("### Navigate through the application:")
    
    st.write(
        "- **Dashboard:** Explore interactive visualizations of player data, including performance metrics, "
        "attendance trends, positional analysis, and injury distribution."
    )

    st.write(
        "- **Machine Learning:** Use predictive models to estimate player performance based on key indicators "
        "such as training attendance, match involvement, and physical condition."
    )

    st.write(
        "- **Data Management:** Manage player records efficiently by adding, updating, or removing data "
        "across all divisions."
    )

    st.write(
        "- **Player Report:** Generate a detailed report for individual players, including performance, "
        "fitness, and development insights."
    )

    st.write("---")

    st.write(
        "This tool is designed to support coaches and club managers in making informed, data-driven decisions "
        "to optimize player development and team performance."
    )

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
            st.session_state.page = "Division Report"





########################################### Dashboard Page ###########################################
# Dashboard Page


elif st.session_state.page == "Dashboard":

    st.title("⚽ Player Performance Dashboard")

    if st.button("Homepage"):
        st.session_state.page = "Home"

    # -------------------------------
    # Key Metrics
    # -------------------------------
    st.subheader("Club Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", len(df))
    col2.metric("Avg Performance", round(df["PerformanceScore"].mean(), 1))
    col3.metric("Avg Attendance", f"{round(df['TrainingAttendanceRate'].mean(),1)}%")

    # -------------------------------
    # Age Distribution
    # -------------------------------
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # -------------------------------
    # Position Distribution
    # -------------------------------
    st.subheader("Player Distribution by Position")
    position_counts = df['Position'].value_counts()
    fig, ax = plt.subplots()
    position_counts.plot(kind='bar', ax=ax, color='lightblue')
    ax.set_title("Players by Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # -------------------------------
    # Goals by Position
    # -------------------------------
    st.subheader("Goals by Position")
    fig, ax = plt.subplots()
    sns.boxplot(x='Position', y='Goals', data=df, ax=ax)
    ax.set_title("Goals Distribution by Position")
    st.pyplot(fig)

    # -------------------------------
    # Performance vs Attendance
    # -------------------------------
    st.subheader("Performance vs Training Attendance")
    fig, ax = plt.subplots()
    sns.scatterplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax)
    sns.regplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax, scatter=False, color='red')
    ax.set_title("Performance vs Attendance")
    st.pyplot(fig)

    # -------------------------------
    # Fitness vs Age
    # -------------------------------
    st.subheader("Fitness vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='FitnessScore', data=df, ax=ax)
    sns.regplot(x='Age', y='FitnessScore', data=df, ax=ax, scatter=False, color='red')
    ax.set_title("Fitness vs Age")
    st.pyplot(fig)

    # -------------------------------
    # Injury Status Distribution
    # -------------------------------
    st.subheader("Injury Status Distribution")
    injury_counts = df['InjuryStatus'].value_counts()
    fig, ax = plt.subplots()
    injury_counts.plot(kind='bar', ax=ax, color='salmon')
    ax.set_title("Injury Status")
    st.pyplot(fig)

    # -------------------------------
    # Violin Plot Age Distribution by Gender
    # -------------------------------
    st.subheader("Age Distribution (Male vs Female)")
    
    fig, ax = plt.subplots()
    
    sns.violinplot(
        x=["All"] * len(df),   # forces a single category
        y="Age",
        hue="Gender",
        data=df,
        split=True,
        inner="quartile",
        ax=ax
    )
    
    ax.set_xlabel("")
    ax.set_title("Age Distribution Split by Gender")
    
    st.pyplot(fig)


########################################### ML Page ###########################################




# Machine Learning Page and title 
elif st.session_state.page == "Machine Learning":
    st.title("Machine Learning (page under construction)")
   

########################################### Data Management Page ###########################################


# Backend Page (Data Management)
elif st.session_state.page == "Data Management":
    # Title for the data management page, emphasizing its purpose
    st.title("Data Input Manager (page udner construction)")


    # Defining spreadsheet and worksheet details
    # These variables ensure the application interacts with the correct Google Sheet and worksheet
    spreadsheet_name = "Dataset"
    worksheet = client.open(spreadsheet_name).sheet1
    headers = worksheet.row_values(1)
    sheet_data = worksheet.get_all_values()


    # Main Page Function
    # Acts as a hub for backend operations, offering options to add, update, or delete employee data
    def main_page():
        st.header("Main Page")
        # Navigational buttons to move between subpages or return to the homepage
        st.write("Choose an action:")
        if st.button("Add New Employee Data"):
            st.session_state.Backend_subpage = "add_employee"
        if st.button("Change Employee Data"):
            st.session_state.Backend_subpage = "change_employee"
        if st.button("Delete Employee Data"):
            st.session_state.Backend_subpage = "delete_employee"
        if st.button("Homepage"):
            st.session_state.page = "Home"

    # Add Employee Page Function
    # Allows users to input details for a new employee, dynamically generating the next Employee Number
    def add_employee_page():
        st.header("Add New Employee Data")
        st.write("Enter details for the new employee:")

        # This section allows to calculate the next Employee Number by finding the max from the max from existing numbers
        employee_numbers = [
            int(row[headers.index("EmployeeNumber")])
            for row in sheet_data[1:]
            if row[headers.index("EmployeeNumber")].isdigit()
        ]
        next_employee_number = max(employee_numbers) + 1

        # This section uses a form for structured data input
        with st.form("add_employee_form"):
            # Capture employee details through various input widgets
            # Widgets are chosen based on data type (e.g., `selectbox` for categories, `slider` for ranges)
            age = st.selectbox("Age", list(range(18, 63)))
            attrition = st.selectbox("Attrition", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
            education = st.slider("Education level", min_value=1, max_value=5)
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

            # If the form is submitted, the new employee's data is added to the Google Sheet
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

        # Navigation buttons to return to the main page or homepage
        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"

    # Change Employee Page Function
    # Allows users to update details for an existing employee
    def change_employee_page():
        st.header("Change Employee Data")
        # Retrieve employee numbers from the dataset
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number", employee_numbers)

        if selected_emp:
            # Locate the employee's row
            emp_index = employee_numbers.index(selected_emp) + 1
            # Fetch current data for the selected employee
            current_data = sheet_data[emp_index]

            with st.form("change_employee_form"):
                # Allow users to edit employee details by pre-filling current values
                updated_employee = [
                    st.text_input(headers[i], value=current_data[i])
                    for i in range(len(headers))
                ]
                if st.form_submit_button("Update"):
                    # Update the row with new data
                    worksheet.update(f"A{emp_index + 1}", [updated_employee])
                    st.success("Employee data updated successfully!")

        # Navigation buttons to return to the main page or homepage
        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"


    # Delete Employee Page Function
    # Allows users to remove an employee's data from the dataset
    def delete_employee_page():
        st.header("Delete Employee Data")
        # Retrieve employee numbers from the dataset
        employee_numbers = [
            row[headers.index("EmployeeNumber")] for row in sheet_data[1:]
        ]
        selected_emp = st.selectbox("Select Employee Number to Delete", employee_numbers)

        if selected_emp:
            # Locate the employee's row
            emp_index = employee_numbers.index(selected_emp) + 1
            if st.button("Delete Employee"):
                # Delete the row from the worksheet
                worksheet.delete_rows(emp_index + 1)
                st.success(f"Employee {selected_emp} deleted successfully!")

        # Navigation buttons to return to the main page or homepage
        if st.button("Previous Page"):
            st.session_state.Backend_subpage = "main"
        if st.button("Homepage"):
            st.session_state.page = "Home"


    # Manage navigation between subpages for adding, updating, or deleting data
    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"


    # Render the appropriate subpage based on user selection
    if st.session_state.Backend_subpage == "main":
        main_page()
    elif st.session_state.Backend_subpage == "add_employee":
        add_employee_page()
    elif st.session_state.Backend_subpage == "change_employee":
        change_employee_page()
    elif st.session_state.Backend_subpage == "delete_employee":
        delete_employee_page()






########################################### PDF Report through GPT ###########################################






# Employee Report Page with title
elif st.session_state.page == "Employee Report":
    st.title("Employee Report")

    # Loads the OpenAI API Key from the secrets in Streamlit 
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
    # If this is not possible it shows the error down below
    except KeyError:
        st.error("OpenAI API Key is missing. Please check your Streamlit secrets.")
        # Button for Homepage
        if st.button("Homepage"):
            # Updates the website back to home
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the try-except code will  stop running
        st.stop()

    # Checking if the EmployeeNumber column is present in the google sheets data from our API. 
    # Although it is expected to always be present, this check is added for safety.
    if "EmployeeNumber" not in df.columns:
        st.error("The 'EmployeeNumber' column is missing from the Google Sheet Data.")
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
    # Creating a dropdown menu for the user to select an employee based on their EmployeeNumber.
    # The options in the dropdown are taken from the "EmployeeNumber" column of the DataFrame.
    # The selected value is stored in the variable "employee_number" for further processing.
    # The method .unique() makes sure that the dropdown contains only unique employee numbers.
    employee_number = st.selectbox("Choose Employee (EmployeeNumber)", df["EmployeeNumber"].unique())

    # Filter data for selected employee from the user and using the selected employee data for the dataframe
    employee_data = df[df["EmployeeNumber"] == employee_number]
     #If the data is empty (to check we used the .empty method) it show the error down below
    if employee_data.empty:
        st.error("The selected EmployeeNumber is not present in the data table.")
        # If the user uses the homebutton you will come back to the homepage and this part of the code will stop running    
        if st.button("Homepage"):
            st.session_state.page = "Home"
        # If you use the homebutton you will come back to the homepage and the currently executed code will  stop running
        st.stop()
    # Extracting the selected entry from the filtered employee_data, while making sure that only the values from first row will be chosen via iloc[]
    employee_data = employee_data.iloc[0]

    #Defining the function used for the report
    # This function is responsible for creating a personalized report for an employee based on the provided data
    def generate_report(employee):
        # Build the prompt for the OpenAI API ChatGPT connection, while only using the given data
        prompt = (
            "Create a short, formal, and professional employee report in English using only the provided data. "
            "The employee does not have a name, so please refer to them by their EmployeeNumber. "
            "Do not add any information not present in the data. Present the information as a cohesive paragraph "
            "without additional speculation.\n\n"
            # The prompt uses the data from the selected employee (number), see down below
            f"Data:\n" #Label to begin here 
            f"EmployeeNumber: {employee['EmployeeNumber']}\n" #Using the EmployeeNumber
            f"Age: {employee['Age']}\n" #Using the Age of the employee
            f"Department: {employee['Department']}\n" #Using the department, in which the employee works
            f"Job Role: {employee['JobRole']}\n" #Using the Job role the employee has
            f"Gender: {employee['Gender']}\n" #Using the Gender the employee has
            f"Education Field: {employee['EducationField']}\n" #Using the Education field the employee has
            f"Years at Company: {employee['YearsAtCompany']}\n" #Using the total amount of working years at the company the employee has
            f"Total Working Years: {employee['TotalWorkingYears']}\n" #Using the total amount of working years in general the employee has
            f"Monthly Income: {employee['MonthlyIncome']}\n" #Using the monthly income the employee has
            f"Business Travel: {employee['BusinessTravel']}\n" #Using the data if the employee is traveling or not
            f"Overtime: {employee['OverTime']}\n" #Using if the employee has Over time
            f"Job Satisfaction (1–4): {employee['JobSatisfaction']}\n" #Using the Job satisfaction of the employee
            f"Work-Life Balance (1–4): {employee['WorkLifeBalance']}\n" #Using the Work life balance of the employee
            f"Relationship Satisfaction (1–4): {employee['RelationshipSatisfaction']}\n" #Using the relationship satisfaction of the employee
            f"Performance Rating: {employee['PerformanceRating']}\n" #Using the performance rating of the employee
            f"Training Times Last Year: {employee['TrainingTimesLastYear']}\n\n" #Using the training times of the employee
            #Final instructions for the OpenAI/ChatGPT and again making sure that ChatGPT is not going to use any additional information
            "Please create a single paragraph that uses only these details, maintains a professional and formal tone, "
            "and does not introduce any additional information beyond what is provided."
        )

        # Asking the OpenAi API/ChatGPT, while using the version of chat gpt 3.5
        # Adding necessary attributes to the message: 
            # Defining a role is mandatory for the OpenAI API. In this case, the role is set to "user" because the prompt represents input from the user
            # Limiting the usage of the tokens, which are used for each report. The tokens are purchased at the OpenAI API website
            # Defining a temperature of 0.0, which means that the model works very deterministically and always gives the most probable answer without adding randomness
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            # Returning the first answer generated by the Open AI API/ChatGPT via .choices[0]
            # While using the parameters of message as content. For this code only the content is relevant the role is just a requirement for the OpenAI API as explained
            # And making sure that everything is well formated via .strip()
            return response.choices[0].message["content"].strip()
        # Handle any exceptions that occur during the API request
        # If an error occurs a detailed error message is displayed using st.error(). and None will be returned, so no report will be available
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None

    # Defines a function to create a PDF report for an employee
    # This function generates a PDF document containing a formatted employee report using the FPDF library
    def create_pdf(report, employee):
        # Creating a new PDF document using the FPDF library assigning, source https://www.youtube.com/watch?v=q70xzDG6nls&list=PLjNQtX45f0dR9K2sMJ5ad9wVjqslNBIC0
        # This line creates a new instance of the FPDF class from the fpdf library, which represents a blank PDF document. The instance is assigned to the variable 'pdf'
        pdf = FPDF()
        #Adding one page to write our report
        pdf.add_page()
        # Set the font for the title of the report
        # Arial is used as the font, "B" indicates bold, and 16 is the font size
        pdf.set_font("Arial", "B", 15)
        # Add the title of the report to the PDF
        # Attributes:
            # width: 0 (spans the whole page)
            # height: 10 (cell height)
            # txt: The title of text, which includes the EmployeeNumber
            # ln: starts a new line after the Title
            # align: C means that it centers the text horizontally
        pdf.cell(0, 10, txt=f"Employee Report (EmployeeNumber: {employee['EmployeeNumber']})", ln=True, align="C")
        # Add vertical spacing after the title, and the new line is implemented 10 units after the title
        pdf.ln(10)
         # The report  is used with a normal style ,Arial, not bold and the font size is set to 12, since it is not a title
        pdf.set_font("Arial", size=11)
        # Add the report text to the PDF as multi-line content
        # Parameters:
            # width: 0 (spans the entire width of the page)
            # height: 10 (line height for each row)
            # txt: Uses the report text from the OpenAI API/ Chat GPT
            # multi_cell automatically wraps the text to fit within the page width
        pdf.multi_cell(0, 10, txt=report)
        # It returns and displays the pdf, which has the ability to be displayed and downloaded
        return pdf

    # Check if the "Generate Report" button is clicked by the user in the Streamlit app.
    if st.button("Generate Report"):
         # Call the generate_report function, passing the selected employee's data (employee_data) as input. The function returns the report generated from the OpenAi APi/ Chat GPT.
        report_text = generate_report(employee_data)
        # Check if the generated report is not empty or None, which ensures that the report is displayed only if it was generated
        if report_text:
            # Adding a subheader in the Streamlit app "Employee Report:"
            st.subheader("Employee Report:")
            # Display the generated employee report from OpenAI API/ ChatGPT in the Streamlit app
            st.write(report_text)

            # Call the create_pdf function to generate a PDF report using the generated report and the employee data. The resulting PDF object is assigned to the variable pdf
            pdf = create_pdf(report_text, employee_data)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            # Creating a download button in streamlit which shows up with the emoji and text "Download PDF" 
            # it uses the data from pdf_bytes
            # Naming the File as EmployeeNumber_1.._Report.pdf
            # mime gives the datatype, in this case pdf, so that the data will be correctly used by the device used from the User 
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"EmployeeNumber_{employee_data['EmployeeNumber']}_Report.pdf",
                mime="application/pdf"
            )

    # Add buttons for navigation back to homepage
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Homepage"):
            st.session_state.page = "Home"
    # Add buttons for navigation back to Dashboard
    with col2:
        if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"

