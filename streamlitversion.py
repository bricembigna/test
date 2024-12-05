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

    # Display 'Back to Home' button
    if st.button("Back to Home"):
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
    st.title("Machine Learning: Predict Monthly Income")
    if st.button("Back to Home"):
        st.session_state.page = "Home"



# Backend Page - Placeholder Content
elif st.session_state.page == "Backend":
    st.title("Backend")
    if st.button("Back to Home"):
        st.session_state.page = "Home"
    st.write("Backend management content will be added here.")
